"""Memory orchestration for semantic profile, episodic history, and working context."""

from __future__ import annotations

import asyncio
import re
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from .schemas import (
    AgentStructuredOutput,
    ConversationTurn,
    ExtractedUserProfile,
    UserPreferences,
    utc_now,
)
from .sqlite_store import SQLiteMemoryStore


class MemoryService:
    """Coordinate durable memory and structured LLM responses."""

    def __init__(self, store: SQLiteMemoryStore | None = None, llm: Any | None = None) -> None:
        self.store = store or SQLiteMemoryStore()
        self.llm = llm

    def _get_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        try:
            from app.gateway import gateway_chat_model
        except ModuleNotFoundError:
            from gateway import gateway_chat_model
        return gateway_chat_model

    async def _structured_invoke(self, schema: type[Any], prompt: str) -> Any:
        model = self._get_llm().with_structured_output(schema)
        if hasattr(model, "ainvoke"):
            return await model.ainvoke(prompt)
        return await asyncio.to_thread(model.invoke, prompt)

    @staticmethod
    def _heuristic_profile(user_id: str, message: str) -> ExtractedUserProfile:
        name_match = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Z][\w -]{1,40})", message, re.IGNORECASE)
        profession_match = re.search(r"\b(?:i work as|my profession is|i am a|i'm a)\s+([^,.!?\n]+)", message, re.IGNORECASE)
        return ExtractedUserProfile(
            name=name_match.group(1).strip() if name_match else None,
            profession=profession_match.group(1).strip() if profession_match else None,
        )

    async def extract_and_update_profile(self, user_id: str, user_message: str) -> UserPreferences:
        """Extract explicitly stated facts and merge them into the semantic profile."""
        profile = self.store.get_user_profile(user_id)
        try:
            extracted: ExtractedUserProfile = await self._structured_invoke(
                ExtractedUserProfile,
                "Extract only facts explicitly stated by the user. Never infer identity or preferences. "
                "Return null for unknown scalar fields.\n\nUser message:\n" + user_message,
            )
        except Exception:
            extracted = self._heuristic_profile(user_id, user_message)

        if extracted.name:
            profile.name = extracted.name
        if extracted.profession:
            profile.profession = extracted.profession
        profile.technical_stack = list(dict.fromkeys(profile.technical_stack + extracted.technical_stack))
        profile.custom_rules = list(dict.fromkeys(profile.custom_rules + extracted.custom_rules))
        profile.last_updated = utc_now()
        self.store.save_user_profile(profile)
        return profile

    def hydrate_working_memory(self, user_id: str, thread_id: str, max_turns: int = 4) -> List[BaseMessage]:
        """Build a bounded LangChain message window from durable semantic/episodic memory."""
        profile = self.store.get_user_profile(user_id)
        thread = self.store.get_thread(thread_id, user_id)
        preference_lines = [
            f"Name: {profile.name}" if profile.name else "",
            f"Profession: {profile.profession}" if profile.profession else "",
            f"Technical stack: {', '.join(profile.technical_stack)}" if profile.technical_stack else "",
            *(f"Preference/rule: {rule}" for rule in profile.custom_rules),
        ]
        header = "Persistent user preferences (use only when relevant):\n" + "\n".join(
            line for line in preference_lines if line
        )
        messages: List[BaseMessage] = [SystemMessage(content=header)]
        for turn in thread.turns[-max_turns:]:
            messages.extend(
                [
                    HumanMessage(content=turn.user_query),
                    AIMessage(content=turn.agent_response),
                ]
            )
        return messages

    @staticmethod
    def _fallback_followups(query: str) -> list[str]:
        topic = query.strip().rstrip("?!. ") or "this topic"
        return [
            f"Can you explain the key details of {topic}?",
            f"What should I do next regarding {topic}?",
        ]

    async def persist_turn(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        response: AgentStructuredOutput,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        followups = response.suggested_followups[:3] or self._fallback_followups(query)
        response.suggested_followups = followups[:3]
        self.store.add_turn_to_thread(
            thread_id,
            user_id,
            ConversationTurn(
                user_query=query,
                agent_response=response.answer,
                suggested_followups=response.suggested_followups,
                metadata=metadata or {},
            ),
        )

    async def chat(self, query: str, config: RunnableConfig) -> AgentStructuredOutput:
        """Run one memory-aware structured chat turn using configurable IDs."""
        configurable = config.get("configurable", {}) if config else {}
        user_id = str(configurable.get("user_id", "default_user"))
        thread_id = str(configurable.get("thread_id", "default_thread"))
        await self.extract_and_update_profile(user_id, query)
        messages = self.hydrate_working_memory(user_id, thread_id)
        messages.append(HumanMessage(content=query))
        prompt = (
            "You are a grounded enterprise assistant. Answer only from the user query and working memory. "
            "Return 2-3 useful suggested follow-up questions. Do not invent personal facts.\n\n"
            + "\n".join(f"{message.type}: {message.content}" for message in messages)
        )
        try:
            result: AgentStructuredOutput = await self._structured_invoke(AgentStructuredOutput, prompt)
        except Exception:
            result = AgentStructuredOutput(
                answer="I was unable to generate a response from the configured model.",
                suggested_followups=self._fallback_followups(query),
            )
        await self.persist_turn(user_id, thread_id, query, result)
        return result


global_memory_service = MemoryService()
