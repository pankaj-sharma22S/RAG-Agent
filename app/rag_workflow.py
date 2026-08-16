"""LangGraph workflow for chat, document RAG, web research, and comparisons."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from ddgs import DDGS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

try:
    from app.llm.gemini import llm
except ModuleNotFoundError:
    # ``uvicorn main:app --app-dir app`` makes ``app`` unavailable as a
    # top-level package, while ``python -m app.rag_workflow`` needs it.
    from llm.gemini import llm


RETRIEVAL_CONFIDENCE_THRESHOLD = 0.65
RETRIEVAL_CHUNK_COUNT = 10
WEB_RESULT_COUNT = 5
LOW_CONFIDENCE_THRESHOLD = 0.35
HIGH_CONFIDENCE_THRESHOLD = 0.70


class EvaluationDecision(BaseModel):
    """Structured decision returned by the evaluator after reviewing chunks."""

    route: Literal["answer", "web", "combined"] = Field(
        description="answer=document chunks only, web=current web only, combined=documents plus web"
    )
    chunk_confidence: float = Field(ge=0, le=1, description="Confidence that chunks answer the request")
    reason: str = Field(description="Short reason for the selected route")
    web_query: str = Field(
        description="Focused current-information search query; empty only for document-only answers"
    )


class UserBiography(BaseModel):
    """LLM-selected long-term facts explicitly shared by the user."""

    facts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Short, stable, useful user facts keyed by a clear lowercase name, such "
            "as name, profession, current_project, preference, skill, or goal."
        ),
    )


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    intent: Literal["chat", "research"]
    retrieval_query: str
    documents: list[Document]
    retrieval_confidence: float
    route: Literal["answer", "web", "combined"]
    evaluator_confidence: float
    evaluator_reason: str
    web_query: str
    web_results: list[dict[str, str]]
    force_document_retrieval: bool
    session_id: str
    biography: dict[str, str]


summary_prompt = PromptTemplate.from_template(
    """Summarize the retrieved document text faithfully. Keep relevant technical
details and do not add facts that are not present in the text.

Document:
{text}

Summary:"""
)

answer_prompt = ChatPromptTemplate.from_template(
    """You are a helpful AI assistant. Answer using only the supplied context.

When both document and web context are supplied, compare them clearly. Treat
web results as current information and cite the relevant web source URLs. Do
not present weak or absent document evidence as fact.

Context:
{context}

User biography (use only if relevant):
{biography}

Question:
{question}

Answer:"""
)


def message_text(message: BaseMessage) -> str:
    """Return message content safely when LangChain content is not a plain string."""
    return message.content if isinstance(message.content, str) else str(message.content)


def conversation_context(state: AgentState, limit: int = 6) -> str:
    messages = state.get("messages", [])[-limit:]
    return "\n".join(f"{message.type}: {message_text(message)}" for message in messages)


def format_biography(biography: dict[str, str] | None) -> str:
    """Format stored, explicitly stated profile facts for the model."""
    if not biography:
        return "No biography details have been shared."
    return "\n".join(f"{key.replace('_', ' ').title()}: {value}" for key, value in biography.items() if value)


def profile_node(state: AgentState) -> dict[str, dict[str, str]]:
    """Extract useful, explicit long-term biography facts without guessing."""
    biography = dict(state.get("biography", {}))
    try:
        extracted = structured_biography.invoke(
            "Analyze the latest user message for important long-term facts that would "
            "help personalize future answers. Examples include their name, profession, "
            "skills, current project, goals, preferences, location, or work status. "
            "Extract only facts explicitly stated by the user; never infer or guess. "
            "Do not retain passwords, API keys, authentication codes, financial account "
            "numbers, government IDs, health information, or other secrets. Return an "
            "empty facts object when nothing useful is stated. Do not overwrite an "
            "existing fact unless the user clearly corrects it.\n\n"
            f"Existing profile:\n{format_biography(biography)}\n\n"
            f"Latest user message:\n{state['question']}"
        )
        for key, value in extracted.facts.items():
            if value.strip():
                biography[key.strip().lower().replace(" ", "_")] = value.strip()
    except Exception:
        # A profile update must never block a normal answer.
        pass
    return {"biography": biography}


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def format_web_results(results: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"Web source: {result['title']}\nURL: {result['url']}\n{result['snippet']}"
        for result in results
    )


summary_chain = RunnableLambda(format_documents) | summary_prompt | llm | StrOutputParser()
answer_chain = answer_prompt | llm | StrOutputParser()
structured_evaluator = llm.with_structured_output(EvaluationDecision)
structured_biography = llm.with_structured_output(UserBiography)


def router_node(state: AgentState) -> dict[str, Literal["chat", "research"]]:
    """Send casual conversation to chat; retrieve chunks for every research task."""
    if state.get("force_document_retrieval"):
        return {"intent": "research"}

    response = llm.invoke(
        "Classify the latest request. Return exactly 'chat' for casual conversation "
        "with no research needed. Return exactly 'research' for requests about user "
        "documents, facts, market/current information, comparisons, or a follow-up "
        "such as 'compare it with the market'. Use conversation history to resolve 'it'.\n\n"
        f"Conversation:\n{conversation_context(state)}\n\nLatest request: {state['question']}"
    )
    return {"intent": "research" if response.content.strip().lower() == "research" else "chat"}


def chat_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    instruction = SystemMessage(
        content=f"Use this biography only when it helps answer the user.\n\n{format_biography(state.get('biography'))}"
    )
    return {"messages": [llm.invoke([instruction, *state["messages"]])]}


def retrieve_documents(state: AgentState) -> dict[str, object]:
    """Retrieve the top relevant chunks before the evaluator chooses a route."""
    try:
        from app.vectorstore.croma_store import get_vectorstore
    except ModuleNotFoundError:
        from vectorstore.croma_store import get_vectorstore

    retrieval_query = f"{state['question']}\n\nConversation context:\n{conversation_context(state)}"
    session_id = state.get("session_id", "").strip()
    if not session_id:
        raise ValueError("A session_id is required to search uploaded documents.")

    matches = get_vectorstore().similarity_search_with_relevance_scores(
        retrieval_query,
        k=RETRIEVAL_CHUNK_COUNT,
        filter={"session_id": session_id},
    )
    documents = [document for document, _score in matches]
    confidence = max((float(score) for _document, score in matches), default=0.0)
    return {
        "retrieval_query": retrieval_query,
        "documents": documents,
        "retrieval_confidence": confidence,
    }


def evaluate_documents(state: AgentState) -> dict[str, object]:
    """Score all retrieved chunks with a schema and select the next graph route."""
    documents = state.get("documents", [])
    retrieval_confidence = state.get("retrieval_confidence", 0.0)
    evidence = format_documents(documents)[:12000] if documents else "No document chunks were retrieved."

    decision = structured_evaluator.invoke(
        "You are the retrieval evaluator. Review the user request, conversation, "
        "retrieved chunks, and vector relevance score. Return the required schema.\n\n"
        "Choose answer only when chunks directly answer the request. Choose web when "
        "chunks are missing, weak, or the request needs current information only. "
        "Choose combined when the user explicitly asks to compare document material "
        "with the current market/web, including follow-ups such as 'compare it with "
        "the market'. For web or combined, make web_query specific using the chunks.\n\n"
        f"User request: {state['question']}\n\nConversation:\n{conversation_context(state)}\n\n"
        f"Vector relevance score: {retrieval_confidence:.3f}\n\nRetrieved chunks:\n{evidence}"
    )

    # Confidence policy: low -> web, medium -> combined, high -> preserve the
    # evaluator's document/current-data decision. This runs before generation.
    confidence = min(retrieval_confidence, decision.chunk_confidence)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        route: Literal["answer", "web", "combined"] = "web"
    elif confidence < HIGH_CONFIDENCE_THRESHOLD:
        route = "combined"
    else:
        route = decision.route
    
    if state.get("force_document_retrieval") and documents:
        route = "answer"

    return {
        "route": route,
        "evaluator_confidence": confidence,
        "evaluator_reason": decision.reason,
        "web_query": decision.web_query.strip() or state["question"],
    }


def web_search_node(state: AgentState) -> dict[str, list[dict[str, str]]]:
    """Search DuckDuckGo with the evaluator's chunk-aware current-data query."""
    try:
        raw_results = DDGS().text(
            state.get("web_query", state["question"]),
            backend="duckduckgo",
            max_results=WEB_RESULT_COUNT,
        )
        return {
            "web_results": [
                {
                    "title": item.get("title", "Untitled result"),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
                for item in raw_results
                if item.get("href") and item.get("body")
            ]
        }
    except Exception:
        return {"web_results": []}


def answer_node(state: AgentState) -> dict[str, list[AIMessage]]:
    context_parts: list[str] = []
    route = state["route"]

    # A document API query always includes its retrieved chunks.  This is a
    # final safeguard if an evaluator selects the web route despite the
    # forced-document setting.
    if state.get("documents") and (
        route in {"answer", "combined"} or state.get("force_document_retrieval")
    ):
        context_parts.append("Indexed document summary:\n" + summary_chain.invoke(state["documents"]))
    if route in {"web", "combined"} and state.get("web_results"):
        context_parts.append("Current web research:\n" + format_web_results(state["web_results"]))

    if not context_parts:
        return {
            "messages": [
                AIMessage(
                    content="I could not find enough reliable document or web evidence to answer this."
                )
            ]
        }

    answer = answer_chain.invoke(
        {
            "context": "\n\n".join(context_parts),
            "question": state["question"],
            "biography": format_biography(state.get("biography")),
        }
    )
    return {"messages": [AIMessage(content=answer)]}


def stream_response(
    question: str,
    messages: list[BaseMessage],
    session_id: str,
    biography: dict[str, str] | None = None,
):
    """Yield the final model response token-by-token for interactive clients.

    The routing and retrieval work remains identical to the graph, while the
    final chat or RAG answer uses the model's streaming interface.
    """
    state: AgentState = {
        "question": question,
        "messages": messages,
        "session_id": session_id,
        "biography": biography if biography is not None else {},
    }
    state.update(profile_node(state))
    if biography is not None:
        biography.clear()
        biography.update(state["biography"])
    state.update(router_node(state))

    if state["intent"] == "chat":
        instruction = SystemMessage(
            content=f"Use this biography only when it helps answer the user.\n\n{format_biography(state.get('biography'))}"
        )
        for chunk in llm.stream([instruction, *messages]):
            content = message_text(chunk)
            if content:
                yield content
        return

    state.update(retrieve_documents(state))
    state.update(evaluate_documents(state))
    if state["route"] in {"web", "combined"}:
        state.update(web_search_node(state))

    context_parts: list[str] = []
    if state.get("documents") and state["route"] in {"answer", "combined"}:
        context_parts.append("Indexed document summary:\n" + summary_chain.invoke(state["documents"]))
    if state["route"] in {"web", "combined"} and state.get("web_results"):
        context_parts.append("Current web research:\n" + format_web_results(state["web_results"]))
    if not context_parts:
        yield "I could not find enough reliable document or web evidence to answer this."
        return

    for chunk in answer_chain.stream(
        {
            "context": "\n\n".join(context_parts),
            "question": question,
            "biography": format_biography(state.get("biography")),
        }
    ):
        if chunk:
            yield str(chunk)


def create_rag_graph():
    """Create the router → retrieve → schema evaluator → answer/web LangGraph."""
    workflow = StateGraph(AgentState)
    workflow.add_node("profile", profile_node)
    workflow.add_node("router", router_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("evaluate", evaluate_documents)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("web_search_for_combined", web_search_node)
    workflow.add_node("answer", answer_node)

    workflow.add_edge(START, "profile")
    workflow.add_edge("profile", "router")
    workflow.add_conditional_edges("router", lambda state: state["intent"], {"chat": "chat", "research": "retrieve"})
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: state["route"],
        {
            "answer": "answer",
            "web": "web_search",
            "combined": "web_search_for_combined",
        },
    )
    workflow.add_edge("web_search", "answer")
    workflow.add_edge("web_search_for_combined", "answer")
    workflow.add_edge("chat", END)
    workflow.add_edge("answer", END)
    return workflow.compile(checkpointer=MemorySaver())


graph_app = create_rag_graph()


if __name__ == "__main__":
    first_turn = True
    print("Ask a question. Type 'exit' to stop.")
    while True:
        query = input("Ask: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        input_messages: list[BaseMessage] = [HumanMessage(content=query)]
        if first_turn:
            input_messages.insert(
                0, SystemMessage(content="You are a helpful AI assistant. Be concise and clear."),
            )
        result = graph_app.invoke(
            {
                "question": query,
                "messages": input_messages,
            },
            config={"configurable": {"thread_id": "user_1"}},
        )
        print(result["messages"][-1].content)
        first_turn = False
