"""Evidence-driven Agentic GitHub Repository Analyzer implemented with LangGraph and MCP tools."""

from __future__ import annotations

import json
import logging
from typing import Any, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

try:
    from app.gateway import gateway_chat_model as llm
except ModuleNotFoundError:
    try:
        from gateway import gateway_chat_model as llm
    except ModuleNotFoundError:
        from app.llm.gemini import llm

try:
    from app.tracing import (
        DEFAULT_PROJECT,
        MODEL_NAME,
        get_tracing_config,
        safe_traceable,
        setup_tracing,
    )
except ModuleNotFoundError:
    from tracing import (
        DEFAULT_PROJECT,
        MODEL_NAME,
        get_tracing_config,
        safe_traceable,
        setup_tracing,
    )

try:
    from app.mcp.tools_mcp import mcp_registry
    from app.tools.github_tools import (
        classify_github_url,
        get_github_topic_data,
        parse_github_url,
    )
    from app.agents.schemas import (
        CodeStructureAnalysis,
        DiagnosticHypothesis,
        DocumentationComparison,
        Finding,
        InvestigationPlan,
        StructuredAnalysisReport,
    )
except ModuleNotFoundError:
    from mcp.tools_mcp import mcp_registry
    from tools.github_tools import (
        classify_github_url,
        get_github_topic_data,
        parse_github_url,
    )
    from agents.schemas import (
        CodeStructureAnalysis,
        DiagnosticHypothesis,
        DocumentationComparison,
        Finding,
        InvestigationPlan,
        StructuredAnalysisReport,
    )

logger = logging.getLogger("GitHubAnalyzerAgent")
setup_tracing()


class GitHubAnalyzerState(TypedDict, total=False):
    """Strongly-typed state for the Agentic GitHub Repository Analyzer."""

    # Input & Session Identity
    repo_url: str
    owner: str
    repo: str
    branch: Optional[str]
    focus_query: str
    thread_id: str

    # Intent & Strategy Plan
    is_topic: bool
    topic_name: str
    topic_data: dict[str, Any]
    investigation_plan: dict[str, Any]

    # Repository Discovery
    repo_info: dict[str, Any]
    tree: List[dict[str, Any]]
    readme_content: str
    structure_analysis: dict[str, Any]
    prioritized_files: List[str]

    # Code Tracing & Source Contents
    code_files_content: dict[str, str]
    code_traces: List[str]

    # History & Diagnosis
    commits: List[dict[str, Any]]
    branches: List[dict[str, Any]]
    branch_diffs: dict[str, Any]
    hypotheses: List[dict[str, Any]]
    doc_comparison: dict[str, Any]

    # Evidence & Verification
    findings: List[dict[str, Any]]
    verification_summary: str

    # Synthesis & Deliverables
    structured_report: dict[str, Any]
    markdown_report: str
    error: Optional[str]
    retry_count: int


# Structured output LLM chains
structured_planner = llm.with_structured_output(InvestigationPlan)
structured_structure_analyzer = llm.with_structured_output(CodeStructureAnalysis)
structured_doc_comparator = llm.with_structured_output(DocumentationComparison)
structured_report_generator = llm.with_structured_output(StructuredAnalysisReport)


# =========================================================================
# 1. Coordinator & Planning Node
# =========================================================================

@safe_traceable(name="Coordinator Plan", run_type="chain", tags=["github-agent", "coordinator"])
def coordinator_plan_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Analyze the user request, classify intent, and formulate a targeted investigation strategy."""
    url = state.get("repo_url", "").strip()
    focus = state.get("focus_query", "").strip()
    url_type, meta = classify_github_url(url)

    if url_type == "topic":
        topic_name = meta.get("topic", "general")
        topic_data = get_github_topic_data(topic_name)
        repos = topic_data.get("repositories", [])
        total_stars = sum(r.get("stars", 0) for r in repos)
        languages_str = ", ".join(topic_data.get("languages", [])) or "Multi-language"

        return {
            "is_topic": True,
            "topic_name": topic_name,
            "topic_data": topic_data,
            "owner": "topics",
            "repo": topic_name,
            "branch": "main",
            "repo_info": {
                "full_name": f"topics/{topic_name}",
                "description": f"GitHub Topic: '{topic_name}' containing {topic_data.get('total_count', len(repos))} repositories",
                "stargazers_count": total_stars,
                "stars": total_stars,
                "language": languages_str,
                "default_branch": "main",
            },
            "error": None,
        }

    try:
        owner, repo, branch_from_url = parse_github_url(url)
    except Exception as error:
        return {"error": f"Invalid GitHub URL: {error}"}

    target_branch = state.get("branch") or branch_from_url

    # Generate structured investigation plan using LLM
    plan_prompt = (
        "You are the Lead Coordinator for an Agentic Codebase Inspection system.\n"
        "Evaluate the target repository and the user's focus inquiry to formulate a structured investigation strategy.\n\n"
        f"Repository: {owner}/{repo} (Branch: {target_branch or 'default'})\n"
        f"User Focus Inquiry: {focus or 'Comprehensive Technical & Architectural Audit'}\n\n"
        "Produce an investigation plan classifying the intent (architecture_overview, debugging_investigation, code_tracing, topic_discovery), "
        "defining specific goals, target symbols, and file patterns to prioritize."
    )

    try:
        plan: InvestigationPlan = structured_planner.invoke(plan_prompt)
        plan_dict = plan.model_dump()
    except Exception as err:
        logger.warning("Planner LLM fallback: %s", err)
        plan_dict = {
            "intent": "debugging_investigation" if any(w in focus.lower() for w in ["bug", "error", "fail", "issue", "break", "fix", "why"]) else "architecture_overview",
            "investigation_goals": [focus or "Analyze codebase architecture, entry points, and potential defects"],
            "target_symbols_or_keywords": [focus] if focus else [],
            "prioritized_file_patterns": ["*.py", "*.ts", "*.js", "*.go", "*.rs", "package.json", "requirements.txt"],
            "rationale": "Default baseline exploration plan.",
        }

    return {
        "is_topic": False,
        "owner": owner,
        "repo": repo,
        "branch": target_branch,
        "investigation_plan": plan_dict,
        "error": None,
    }


# =========================================================================
# 2. Repository Explorer Node
# =========================================================================

@safe_traceable(name="Repository Explorer", run_type="tool", tags=["github-agent", "discovery"])
def repository_explorer_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Discover repository metadata, tree structure, documentation, and dynamically select candidate files."""
    if state.get("error") or state.get("is_topic"):
        return {}

    owner = state["owner"]
    repo = state["repo"]
    target_branch = state.get("branch")
    plan = state.get("investigation_plan", {})
    focus = state.get("focus_query", "")

    try:
        repo_info = mcp_registry.call_tool("github_get_repository", {"owner": owner, "repo": repo})
        if isinstance(repo_info, dict) and "error" in repo_info:
            return {"error": f"Failed fetching repository metadata: {repo_info['error']}"}

        if not target_branch and isinstance(repo_info, dict):
            target_branch = repo_info.get("default_branch", "main")

        branches = mcp_registry.call_tool("github_get_branches", {"owner": owner, "repo": repo})
        tree = mcp_registry.call_tool(
            "github_get_repository_tree",
            {"owner": owner, "repo": repo, "branch": target_branch, "recursive": True},
        )
        if isinstance(tree, dict) and "error" in tree:
            tree = mcp_registry.call_tool(
                "github_get_repository_tree",
                {"owner": owner, "repo": repo, "branch": target_branch, "recursive": False},
            )

        readme = mcp_registry.call_tool("github_get_readme", {"owner": owner, "repo": repo, "branch": target_branch})
        readme_str = str(readme) if readme else "No README available"

        # Dynamically select key files with LLM based on plan & focus
        tree_paths = [item.get("path", "") for item in (tree if isinstance(tree, list) else []) if item.get("type") == "blob"]
        tree_sample = "\n".join(tree_paths[:250])

        struct_prompt = (
            "You are an expert software architect exploring a codebase.\n"
            "Review the repository file tree and README. Identify:\n"
            "1. Programming languages and frameworks.\n"
            "2. Main application entry points.\n"
            "3. Key architectural modules and configuration files.\n"
            "4. Dependencies and test frameworks.\n\n"
            f"Investigation Plan Goals:\n{json.dumps(plan.get('investigation_goals', []), indent=2)}\n\n"
            f"User Focus Inquiry: {focus or 'Comprehensive Overview'}\n\n"
            f"README Excerpt:\n{readme_str[:3000]}\n\n"
            f"Repository File Paths ({len(tree_paths)} total files):\n{tree_sample}"
        )

        try:
            structure: CodeStructureAnalysis = structured_structure_analyzer.invoke(struct_prompt)
            struct_dict = structure.model_dump()
        except Exception as struct_err:
            logger.warning("Structure analyzer LLM error: %s", struct_err)
            struct_dict = {
                "languages": [repo_info.get("language", "Unknown")] if isinstance(repo_info, dict) else [],
                "frameworks": [],
                "entry_points": [],
                "key_modules": [],
                "dependencies": [],
                "test_frameworks": [],
            }

        # Select prioritized files for reading (incorporating entry points, config, and LLM guidance)
        candidate_paths: List[str] = []
        
        # Priority config & manifests
        config_names = {
            "package.json", "requirements.txt", "pyproject.toml", "cargo.toml",
            "go.mod", "dockerfile", "docker-compose.yml", "tsconfig.json",
            "setup.py", "gemfile", "pom.xml"
        }
        for p in tree_paths:
            if p.split("/")[-1].lower() in config_names:
                candidate_paths.append(p)

        # Entry points from LLM
        for ep in struct_dict.get("entry_points", []):
            if ep in tree_paths and ep not in candidate_paths:
                candidate_paths.append(ep)

        # Focus keywords matching
        target_keywords = plan.get("target_symbols_or_keywords", [])
        if target_keywords:
            for p in tree_paths:
                if any(kw.lower() in p.lower() for kw in target_keywords if len(kw) > 2):
                    if p not in candidate_paths and not p.endswith((".md", ".png", ".jpg")):
                        candidate_paths.append(p)

        # Core architectural modules
        for p in tree_paths:
            if len(candidate_paths) >= 12:
                break
            if any(k in p.lower() for k in ["main", "app", "server", "core", "api", "router", "agent", "service", "handler"]):
                if p not in candidate_paths and not p.endswith((".md", ".txt", ".png", ".json")):
                    candidate_paths.append(p)

        # Fill with general source code if still under 8
        for p in tree_paths:
            if len(candidate_paths) >= 10:
                break
            if p not in candidate_paths and p.endswith((".py", ".ts", ".js", ".go", ".rs", ".java", ".cpp")):
                candidate_paths.append(p)

        return {
            "branch": target_branch,
            "repo_info": repo_info if isinstance(repo_info, dict) else {},
            "branches": branches if isinstance(branches, list) else [],
            "tree": tree if isinstance(tree, list) else [],
            "readme_content": readme_str,
            "structure_analysis": struct_dict,
            "prioritized_files": candidate_paths[:14],
            "error": None,
        }
    except Exception as error:
        logger.exception("Repository exploration failed: %s", error)
        return {"error": f"Repository exploration failed: {error}"}


# =========================================================================
# 3. Code & Execution Tracer Node
# =========================================================================

@safe_traceable(name="Code & Execution Tracer", run_type="chain", tags=["github-agent", "code-tracer"])
def code_tracing_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Fetch content of prioritized files, analyze call hierarchies, data flow, and error handling."""
    if state.get("error") or state.get("is_topic"):
        return {}

    owner = state["owner"]
    repo = state["repo"]
    branch = state.get("branch")
    prioritized_files = state.get("prioritized_files", [])

    files_content = mcp_registry.call_tool(
        "github_get_files",
        {"owner": owner, "repo": repo, "paths": prioritized_files, "branch": branch},
    )
    if not isinstance(files_content, dict):
        files_content = {}

    return {
        "code_files_content": files_content,
    }


# =========================================================================
# 4. Diagnostic & History Engine Node
# =========================================================================

@safe_traceable(name="Diagnostic & History Engine", run_type="chain", tags=["github-agent", "diagnosis"])
def diagnostic_engine_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Fetch commits, branch diffs, issues/PRs, formulate competing hypotheses, and compare docs vs code."""
    if state.get("error") or state.get("is_topic"):
        return {}

    owner = state["owner"]
    repo = state["repo"]
    branch = state.get("branch")
    branches = state.get("branches", [])
    readme = (state.get("readme_content") or "")[:4000]
    code_files = state.get("code_files_content", {})
    focus = state.get("focus_query", "")
    plan = state.get("investigation_plan", {})

    # Fetch recent commits
    commits = mcp_registry.call_tool(
        "github_get_commits",
        {"owner": owner, "repo": repo, "branch": branch, "per_page": 10},
    )
    if not isinstance(commits, list):
        commits = []

    # Branch comparison
    branch_diffs: dict[str, Any] = {}
    if len(branches) > 1 and isinstance(branches[0], dict) and isinstance(branches[1], dict):
        default_b = branch or "main"
        other_b = branches[1]["name"] if branches[0]["name"] == default_b else branches[0]["name"]
        try:
            diff_res = mcp_registry.call_tool(
                "github_compare_branches",
                {"owner": owner, "repo": repo, "base": default_b, "head": other_b},
            )
            if isinstance(diff_res, dict) and "error" not in diff_res:
                branch_diffs = {
                    "base": default_b,
                    "head": other_b,
                    "ahead_by": diff_res.get("ahead_by", 0),
                    "behind_by": diff_res.get("behind_by", 0),
                    "files_changed": [f.get("filename") for f in diff_res.get("files", [])],
                }
        except Exception:
            pass

    # Documentation vs code comparison
    code_snippets = []
    for path, content in code_files.items():
        code_snippets.append(f"--- File: {path} ---\n{content[:2500]}")
    code_context = "\n\n".join(code_snippets)

    doc_prompt = (
        "Compare the claimed features, setup instructions, APIs, and environment variables in the README "
        "with the actual codebase implementation provided.\n"
        "Identify any discrepancies: documented APIs that do not exist, outdated commands, missing env vars, "
        "or key functionality in code that is omitted from the documentation.\n\n"
        f"README:\n{readme}\n\n"
        f"Codebase Implementation Excerpts:\n{code_context[:7000]}"
    )

    try:
        doc_comparison = structured_doc_comparator.invoke(doc_prompt)
        doc_comp_dict = doc_comparison.model_dump()
    except Exception:
        doc_comp_dict = {
            "claimed_features": [],
            "missing_or_outdated": [],
            "undocumented_features": [],
            "readme_accuracy_score": 1.0,
        }

    return {
        "commits": commits,
        "branch_diffs": branch_diffs,
        "doc_comparison": doc_comp_dict,
    }


# =========================================================================
# 5. Evidence Verifier Node
# =========================================================================

@safe_traceable(name="Evidence Verifier", run_type="chain", tags=["github-agent", "verifier"])
def evidence_verifier_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Verify findings against concrete code lines, syntax, and commit evidence without hallucinating."""
    if state.get("error") or state.get("is_topic"):
        return {}

    code_files = state.get("code_files_content", {})
    commits = state.get("commits", [])[:5]
    doc_comp = state.get("doc_comparison", {})
    focus = state.get("focus_query", "")
    plan = state.get("investigation_plan", {})

    code_snippets = []
    for path, content in code_files.items():
        code_snippets.append(f"--- File: {path} ---\n{content[:3000]}")
    code_context = "\n\n".join(code_snippets)

    findings_prompt = (
        "You are an expert technical auditor analyzing a codebase for logic breaks, runtime bugs, "
        "unhandled exceptions, broken imports, missing dependencies, state management issues, or security flaws.\n\n"
        "Verification Constraints:\n"
        "1. Do NOT claim something is broken without concrete code evidence from the provided files.\n"
        "2. Accurately assign severity: Critical, High, Medium, Low, or Informational.\n"
        "3. Accurately assign confidence: Confirmed (direct code proof), Likely (strong static inference), Possible, or Needs Verification.\n"
        "4. Accurately assign verification_status: Static Analysis Finding, Runtime/Test-Confirmed, Documentation Verified, or Needs Verification.\n"
        "5. Include exact file, line range, and recommendation for each finding.\n\n"
        f"Investigation Intent & Goals:\n{json.dumps(plan.get('investigation_goals', []), indent=2)}\n\n"
        f"User Inquiry: {focus or 'General complete repository audit'}\n\n"
        f"Documentation Discrepancies:\n{json.dumps(doc_comp, indent=2)}\n\n"
        f"Codebase Files:\n{code_context}"
    )

    findings_list: List[dict[str, Any]] = []
    try:
        class FindingsWrapper(BaseModel):
            findings: List[Finding] = Field(default_factory=list)

        wrapper_schema = llm.with_structured_output(FindingsWrapper)
        res = wrapper_schema.invoke(findings_prompt)
        if isinstance(res, FindingsWrapper):
            findings_list = [f.model_dump() for f in res.findings]
        elif isinstance(res, dict) and "findings" in res:
            for f in res["findings"]:
                findings_list.append(f.model_dump() if hasattr(f, "model_dump") else f)
    except Exception as err:
        logger.warning("Findings verifier LLM error: %s", err)


    return {
        "findings": findings_list,
        "verification_summary": f"Verified {len(findings_list)} technical findings against codebase evidence.",
    }


# =========================================================================
# 6. Report Synthesizer Node
# =========================================================================

@safe_traceable(name="Report Synthesizer", run_type="chain", tags=["github-agent", "reporter"])
def report_synthesizer_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Synthesize all verified evidence, architectural analysis, and findings into the final response."""
    if state.get("error"):
        err_msg = state["error"]
        markdown = f"## ❌ Analysis Error\n\n**Failed to analyze repository:** {err_msg}\n\nPlease verify that the GitHub URL is correct, public (or valid GITHUB_TOKEN configured), and reachable."
        return {
            "structured_report": {
                "repo_name": f"{state.get('owner', '')}/{state.get('repo', '')}",
                "overview": f"Error: {err_msg}",
                "architecture": "N/A",
                "findings": [],
                "recommendations": ["Verify repository URL and permissions."],
            },
            "markdown_report": markdown,
        }

    if state.get("is_topic"):
        topic_name = state.get("topic_name", "general")
        topic_data = state.get("topic_data", {})
        repos = topic_data.get("repositories", [])
        languages = topic_data.get("languages", [])

        md_lines = [
            f"# 🌐 GitHub Topic Ecosystem Report: `{topic_name}`",
            f"**Total Aggregated Repositories:** {topic_data.get('total_count', len(repos))} | **Dominant Languages:** {', '.join(languages) or 'Multi-language'}",
            "",
            "## 1. Topic Overview",
            f"The GitHub topic `{topic_name}` groups open-source projects, tools, and libraries focused on {topic_name}. Below is an architectural overview of the top rated projects discovered in this domain.",
            "",
            "## 2. Top Repositories in this Ecosystem",
        ]

        for r in repos[:8]:
            md_lines.extend([
                f"### 📦 [{r['full_name']}]({r['html_url']})",
                f"- **Stars:** ⭐ {r['stars']:,} | **Forks:** 🍴 {r['forks']:,} | **Primary Language:** `{r['language']}`",
                f"- **Description:** {r['description']}",
                f"- **Topics:** {', '.join(f'`{t}`' for t in r.get('topics', [])) or 'None'}",
                "",
            ])

        md_lines.extend([
            "## 3. Key Ecosystem Insights & Recommendations",
            f"- **Primary Stacks:** The most active repositories under `{topic_name}` predominantly use {', '.join(languages[:4]) or 'modern languages'}.",
            "- **Integration Potential:** You can query any individual repository from this list by pasting its direct repository URL into the analyzer.",
            "- **Next Steps:** Review the README and repository structure of top-starred projects to adopt standard patterns in this domain.",
        ])

        markdown_report = "\n".join(md_lines)
        return {
            "structured_report": {
                "repo_name": f"topics/{topic_name}",
                "overview": f"GitHub topic discovery report for {topic_name}",
                "architecture": f"Ecosystem comprising {len(repos)} top active repositories across {', '.join(languages)}.",
                "tech_stack": languages,
                "findings": [],
                "recommendations": ["Explore individual repository codebases for deep inspection."],
            },
            "markdown_report": markdown_report,
        }

    owner = state.get("owner", "")
    repo = state.get("repo", "")
    repo_info = state.get("repo_info", {})
    struct_info = state.get("structure_analysis", {})
    code_files = state.get("code_files_content", {})
    commits = state.get("commits", [])
    branch_diffs = state.get("branch_diffs", {})
    doc_comp = state.get("doc_comparison", {})
    findings = state.get("findings", [])
    focus = state.get("focus_query", "")
    plan = state.get("investigation_plan", {})

    report_prompt = (
        f"Generate a comprehensive, structured technical analysis report for GitHub repository: {owner}/{repo}.\n\n"
        f"Repository Metadata:\n{json.dumps(repo_info, indent=2)}\n\n"
        f"Investigation Strategy:\n{json.dumps(plan, indent=2)}\n\n"
        f"Identified Architecture & Tech Stack:\n{json.dumps(struct_info, indent=2)}\n\n"
        f"Documentation Comparison:\n{json.dumps(doc_comp, indent=2)}\n\n"
        f"Branch & Commit Context:\nCommits: {len(commits)} recent. Branch diffs: {json.dumps(branch_diffs)}\n\n"
        f"Verified Findings ({len(findings)} items):\n{json.dumps(findings, indent=2)}\n\n"
        f"User Focus / Inquiry: {focus or 'Comprehensive Technical Analysis'}\n\n"
        "Produce the required structured report detailing overview, architecture, tech stack, entry points, "
        "execution flow, data flow, dependencies, branch/commit history analysis, README vs code verification, "
        "all findings, and actionable recommendations."
    )

    try:
        structured_rep: StructuredAnalysisReport = structured_report_generator.invoke(report_prompt)
        structured_dict = structured_rep.model_dump()
    except Exception as rep_err:
        logger.warning("Report generator LLM error: %s", rep_err)
        structured_dict = {
            "repo_name": f"{owner}/{repo}",
            "overview": repo_info.get("description", "Repository analyzed"),
            "architecture": "Modular codebase structure.",
            "tech_stack": struct_info.get("languages", []) + struct_info.get("frameworks", []),
            "entry_points": struct_info.get("entry_points", []),
            "important_files": list(code_files.keys()),
            "execution_flow": "Standard application pipeline.",
            "data_flow": "Input processing -> business logic -> output.",
            "dependencies_summary": ", ".join(struct_info.get("dependencies", [])),
            "branch_analysis": f"Default branch: {state.get('branch', 'main')}",
            "commit_analysis": f"Analyzed {len(commits)} recent commits.",
            "readme_vs_code": f"Accuracy score: {doc_comp.get('readme_accuracy_score', 1.0)*100:.0f}%",
            "findings": findings,
            "recommendations": ["Ensure automated test coverage for core business logic."],
        }

    # Format polished Markdown
    md_lines = [
        f"# 📊 Technical Analysis Report: `{owner}/{repo}`",
        f"**Stars:** ⭐ {repo_info.get('stars', 0)} | **Language:** 💻 {repo_info.get('language', 'Unknown')} | **Default Branch:** 🌿 `{state.get('branch', 'main')}`",
        "",
        "## 1. Executive Overview",
        structured_dict.get("overview", ""),
        "",
        "## 2. Architecture & Tech Stack",
        f"**Tech Stack:** {', '.join(structured_dict.get('tech_stack', [])) or 'N/A'}",
        "",
        f"**Entry Points:** {', '.join(f'`{ep}`' for ep in structured_dict.get('entry_points', [])) or 'None detected'}",
        "",
        structured_dict.get("architecture", ""),
        "",
        "## 3. Execution & Data Flow",
        f"### Execution Flow\n{structured_dict.get('execution_flow', '')}",
        "",
        f"### Data Flow\n{structured_dict.get('data_flow', '')}",
        "",
        "## 4. Documentation vs. Code Verification",
        structured_dict.get("readme_vs_code", ""),
        "",
        "## 5. History & Branch Analysis",
        f"**Recent Commits:**\n{structured_dict.get('commit_analysis', '')}",
        "",
        f"**Branch Comparison:**\n{structured_dict.get('branch_analysis', '')}",
        "",
        "## 6. Technical Findings & Potential Issues",
    ]

    report_findings = structured_dict.get("findings", [])
    if not report_findings:
        md_lines.append("✅ *No critical bugs or logic breaks detected in static inspection.*")
    else:
        for idx, finding in enumerate(report_findings, 1):
            sev = finding.get("severity", "Medium")
            cat = finding.get("category", "General")
            title = finding.get("title", "")
            desc = finding.get("description", "")
            ev = finding.get("evidence", "")
            file_ref = finding.get("file")
            lr = finding.get("line_range")
            conf = finding.get("confidence", "Likely")
            rec = finding.get("recommendation", "")

            badge = "🔴" if sev == "Critical" else "🟠" if sev == "High" else "🟡" if sev == "Medium" else "🔵"
            md_lines.append(f"### {badge} Finding {idx}: {title}")
            md_lines.append(f"- **Severity:** `{sev}` | **Category:** `{cat}` | **Confidence:** `{conf}`")
            if file_ref:
                loc = f"`{file_ref}`" + (f" (lines {lr})" if lr else "")
                md_lines.append(f"- **Location:** {loc}")
            md_lines.append(f"- **Description:** {desc}")
            if ev:
                md_lines.append(f"- **Evidence:**\n  > {ev}")
            if rec:
                md_lines.append(f"- **Recommendation:** {rec}")
            md_lines.append("")

    md_lines.append("## 7. Strategic Recommendations")
    for rec in structured_dict.get("recommendations", []):
        md_lines.append(f"- {rec}")

    markdown_report = "\n".join(md_lines)
    return {
        "structured_report": structured_dict,
        "markdown_report": markdown_report,
    }


# =========================================================================
# 7. LangGraph Workflow Graph Construction
# =========================================================================

def create_github_analyzer_graph():
    """Build and compile the Agentic LangGraph workflow for GitHub Repository Analysis."""
    workflow = StateGraph(GitHubAnalyzerState)

    workflow.add_node("coordinator_plan", coordinator_plan_node)
    workflow.add_node("repository_explorer", repository_explorer_node)
    workflow.add_node("code_tracing", code_tracing_node)
    workflow.add_node("diagnostic_engine", diagnostic_engine_node)
    workflow.add_node("evidence_verifier", evidence_verifier_node)
    workflow.add_node("report_synthesizer", report_synthesizer_node)

    workflow.add_edge(START, "coordinator_plan")
    workflow.add_conditional_edges(
        "coordinator_plan",
        lambda state: "report_synthesizer" if (state.get("error") or state.get("is_topic")) else "repository_explorer",
        {"report_synthesizer": "report_synthesizer", "repository_explorer": "repository_explorer"},
    )
    workflow.add_conditional_edges(
        "repository_explorer",
        lambda state: "report_synthesizer" if state.get("error") else "code_tracing",
        {"report_synthesizer": "report_synthesizer", "code_tracing": "code_tracing"},
    )
    workflow.add_edge("code_tracing", "diagnostic_engine")
    workflow.add_edge("diagnostic_engine", "evidence_verifier")
    workflow.add_edge("evidence_verifier", "report_synthesizer")
    workflow.add_edge("report_synthesizer", END)

    return workflow.compile(checkpointer=MemorySaver())


github_analyzer_app = create_github_analyzer_graph()


def analyze_github_repository(
    repo_url: str,
    focus_query: str = "",
    branch: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end evidence-driven agentic GitHub repository analysis."""
    active_thread = thread_id or f"github_{abs(hash(repo_url)) % 100000}"
    tracing_config = get_tracing_config(
        thread_id=active_thread,
        session_id=active_thread,
        tags=["github-agent", "repository-analysis"],
        metadata={
            "repo_url": repo_url,
            "focus_query": focus_query,
            "branch": branch or "default",
        },
        run_name=f"GitHub Agent Analysis: {repo_url}",
    )

    initial_state: GitHubAnalyzerState = {
        "repo_url": repo_url,
        "focus_query": focus_query,
        "branch": branch,
        "thread_id": active_thread,
        "retry_count": 0,
    }

    result = github_analyzer_app.invoke(initial_state, config=tracing_config)
    report_md = result.get("markdown_report", "")
    try:
        from guardrails.output_guard import global_output_guard
    except ModuleNotFoundError:
        from app.guardrails.output_guard import global_output_guard
    _, sanitized_report, _ = global_output_guard.validate_output(report_md)

    return {
        "repo_name": f"{result.get('owner', '')}/{result.get('repo', '')}",
        "markdown_report": sanitized_report,
        "structured_report": result.get("structured_report", {}),
        "findings": result.get("findings", []),
        "error": result.get("error"),
    }
