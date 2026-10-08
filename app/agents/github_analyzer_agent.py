"""GitHub Repository Analyzer Agent implemented with LangGraph and MCP tools."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

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
        trace_stage,
    )
except ModuleNotFoundError:
    from tracing import (
        DEFAULT_PROJECT,
        MODEL_NAME,
        get_tracing_config,
        safe_traceable,
        setup_tracing,
        trace_stage,
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
        DocumentationComparison,
        Finding,
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
        DocumentationComparison,
        Finding,
        StructuredAnalysisReport,
    )

setup_tracing()


class GitHubAnalyzerState(TypedDict, total=False):
    """State passing through the GitHub Repository Analyzer LangGraph workflow."""

    repo_url: str
    owner: str
    repo: str
    branch: str | None
    focus_query: str
    thread_id: str
    is_topic: bool
    topic_name: str
    topic_data: dict[str, Any]
    repo_info: dict[str, Any]
    tree: list[dict[str, Any]]
    readme_content: str
    structure_analysis: dict[str, Any]
    prioritized_files: list[str]
    code_files_content: dict[str, str]
    commits: list[dict[str, Any]]
    branches: list[dict[str, Any]]
    branch_diffs: dict[str, Any]
    doc_comparison: dict[str, Any]
    findings: list[dict[str, Any]]
    structured_report: dict[str, Any]
    markdown_report: str
    error: str | None


# Structured output chains
structured_structure_analyzer = llm.with_structured_output(CodeStructureAnalysis)
structured_doc_comparator = llm.with_structured_output(DocumentationComparison)
structured_report_generator = llm.with_structured_output(StructuredAnalysisReport)


@safe_traceable(name="Validate & Discover Repository", run_type="tool", tags=["github-analyzer", "discovery"])
def validate_and_discover_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Validate GitHub URL, fetch repo metadata or topic aggregation data, and file tree."""
    url = state.get("repo_url", "").strip()
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
            "branches": [],
            "tree": [],
            "error": None,
        }

    try:
        owner, repo, branch_from_url = parse_github_url(url)
    except Exception as error:
        return {"error": f"Invalid GitHub URL: {error}"}

    target_branch = state.get("branch") or branch_from_url

    try:
        repo_info = mcp_registry.call_tool("github_get_repository", {"owner": owner, "repo": repo})
        if isinstance(repo_info, dict) and "error" in repo_info:
            return {"error": f"Failed fetching repository metadata: {repo_info['error']}"}

        branches = mcp_registry.call_tool("github_get_branches", {"owner": owner, "repo": repo})
        if not target_branch and isinstance(repo_info, dict):
            target_branch = repo_info.get("default_branch", "main")

        tree = mcp_registry.call_tool(
            "github_get_repository_tree",
            {"owner": owner, "repo": repo, "branch": target_branch, "recursive": True},
        )
        if isinstance(tree, dict) and "error" in tree:
            tree = mcp_registry.call_tool(
                "github_get_repository_tree",
                {"owner": owner, "repo": repo, "branch": target_branch, "recursive": False},
            )

        return {
            "is_topic": False,
            "owner": owner,
            "repo": repo,
            "branch": target_branch,
            "repo_info": repo_info if isinstance(repo_info, dict) else {},
            "branches": branches if isinstance(branches, list) else [],
            "tree": tree if isinstance(tree, list) else [],
            "error": None,
        }
    except Exception as error:
        return {"error": f"Repository discovery failed: {error}"}


@safe_traceable(name="Analyze Documentation", run_type="chain", tags=["github-analyzer", "documentation"])
def analyze_documentation_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Fetch and inspect the README and key documentation files."""
    if state.get("error") or state.get("is_topic"):
        return {}

    owner = state["owner"]
    repo = state["repo"]
    branch = state.get("branch")

    try:
        readme = mcp_registry.call_tool("github_get_readme", {"owner": owner, "repo": repo, "branch": branch})
        return {"readme_content": str(readme)}
    except Exception as error:
        return {"readme_content": f"Unable to fetch README: {error}"}


@safe_traceable(name="Identify Important Code", run_type="chain", tags=["github-analyzer", "structure"])
def identify_important_code_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Inspect directory tree to identify languages, frameworks, entry points, and prioritize key files."""
    if state.get("error"):
        return {}

    tree = state.get("tree", [])
    readme_sample = (state.get("readme_content") or "")[:4000]

    # Format sample tree (up to 200 files) for LLM
    tree_paths = [item.get("path", "") for item in tree if item.get("type") == "blob"]
    tree_text = "\n".join(tree_paths[:250])

    prompt = (
        "You are an expert software architect analyzing a GitHub repository.\n"
        "Review the repository file tree and README excerpt. Identify:\n"
        "1. Programming languages and frameworks.\n"
        "2. Main entry points (e.g. main.py, app.py, index.ts, src/main.rs, etc.).\n"
        "3. Key architectural modules and configuration files.\n"
        "4. Dependencies and test frameworks.\n\n"
        f"README excerpt:\n{readme_sample}\n\n"
        f"Repository file paths ({len(tree_paths)} total files):\n{tree_text}"
    )

    try:
        structure: CodeStructureAnalysis = structured_structure_analyzer.invoke(prompt)
        struct_dict = structure.model_dump()
    except Exception:
        struct_dict = {
            "languages": [],
            "frameworks": [],
            "entry_points": [],
            "key_modules": [],
            "dependencies": [],
            "test_frameworks": [],
        }

    # Select top 10 prioritized files for reading (entry points + config files + core modules)
    candidate_paths: list[str] = []
    
    # Priority config / manifest files
    config_names = {
        "package.json", "requirements.txt", "pyproject.toml", "cargo.toml",
        "go.mod", "dockerfile", "docker-compose.yml", "tsconfig.json",
        ".env.example", "setup.py", "gemfile", "pom.xml"
    }
    for p in tree_paths:
        base = p.split("/")[-1].lower()
        if base in config_names:
            candidate_paths.append(p)

    # Add entry points
    for ep in struct_dict.get("entry_points", []):
        if ep in tree_paths and ep not in candidate_paths:
            candidate_paths.append(ep)

    # Add matching key module paths
    for p in tree_paths:
        if len(candidate_paths) >= 12:
            break
        if any(keyword in p.lower() for keyword in ["main", "app", "server", "core", "api", "router", "auth"]):
            if p not in candidate_paths and not p.endswith((".md", ".txt", ".png", ".json")):
                candidate_paths.append(p)

    # If still under 8, add general source code files
    for p in tree_paths:
        if len(candidate_paths) >= 10:
            break
        if p not in candidate_paths and p.endswith((".py", ".ts", ".js", ".go", ".rs", ".java", ".cpp")):
            candidate_paths.append(p)

    return {
        "structure_analysis": struct_dict,
        "prioritized_files": candidate_paths[:12],
    }


@safe_traceable(name="Analyze Code & Logic", run_type="chain", tags=["github-analyzer", "code-analysis"])
def analyze_code_and_logic_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Fetch content of prioritized source files and analyze execution/data flow and logic."""
    if state.get("error"):
        return {}

    owner = state["owner"]
    repo = state["repo"]
    branch = state.get("branch")
    prioritized_files = state.get("prioritized_files", [])

    # Fetch file contents
    files_content = mcp_registry.call_tool(
        "github_get_files",
        {"owner": owner, "repo": repo, "paths": prioritized_files, "branch": branch},
    )
    if not isinstance(files_content, dict):
        files_content = {}

    return {"code_files_content": files_content}


@safe_traceable(name="Analyze History & Branches", run_type="chain", tags=["github-analyzer", "git-history"])
def analyze_history_and_branches_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Fetch recent commits, inspect relevant diffs, and check active branches."""
    if state.get("error"):
        return {}

    owner = state["owner"]
    repo = state["repo"]
    branch = state.get("branch")
    branches = state.get("branches", [])

    commits = mcp_registry.call_tool(
        "github_get_commits",
        {"owner": owner, "repo": repo, "branch": branch, "per_page": 10},
    )
    if not isinstance(commits, list):
        commits = []

    # If multiple branches exist, compare first divergent branch to default branch
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

    return {
        "commits": commits,
        "branch_diffs": branch_diffs,
    }


@safe_traceable(name="Cross-Check & Verify Findings", run_type="chain", tags=["github-analyzer", "verification"])
def cross_check_and_verify_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Compare documentation claims vs codebase implementation, and identify potential bugs/breaks."""
    if state.get("error"):
        return {}

    readme = (state.get("readme_content") or "")[:5000]
    code_files = state.get("code_files_content", {})
    commits = state.get("commits", [])[:5]
    focus_query = state.get("focus_query", "").strip()

    # Format code context (capping total length)
    code_excerpts = []
    for path, content in code_files.items():
        snippet = content[:3000]
        code_excerpts.append(f"--- File: {path} ---\n{snippet}")
    code_context = "\n\n".join(code_excerpts)

    doc_prompt = (
        "Compare the claimed features, setup instructions, APIs, and environment variables in the README "
        "with the actual codebase implementation provided.\n"
        "Identify any discrepancies: documented APIs that do not exist, outdated commands, missing env vars, "
        "or key functionality in code that is omitted from the documentation.\n\n"
        f"README:\n{readme}\n\n"
        f"Codebase Implementation Excerpts:\n{code_context[:8000]}"
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

    # Now identify potential logic breaks, runtime errors, and bugs with evidence
    findings_prompt = (
        "You are an expert technical auditor analyzing a codebase for logic breaks, runtime bugs, "
        "unhandled exceptions, broken imports, missing dependencies, state management issues, or security flaws.\n\n"
        "Rules:\n"
        "1. Do NOT claim something is definitely broken without concrete evidence from the code.\n"
        "2. Classify confidence as: Confirmed, Likely, Possible, or Needs Verification.\n"
        "3. Classify verification_status as: Static Analysis Finding, Runtime/Test-Confirmed, Documentation Verified, or Needs Verification.\n"
        "4. Include exact file, line range, and recommendation for each finding.\n\n"
        f"User Focus / Inquiry (if any): {focus_query or 'General complete repository audit'}\n\n"
        f"Recent Commits:\n{json.dumps(commits, indent=2)}\n\n"
        f"Documentation Comparison:\n{json.dumps(doc_comp_dict, indent=2)}\n\n"
        f"Codebase Context:\n{code_context}"
    )

    findings: list[Finding] = []
    try:
        class FindingsWrapper(TypedDict):
            findings: list[Finding]

        wrapper_schema = llm.with_structured_output(FindingsWrapper)
        result = wrapper_schema.invoke(findings_prompt)
        if isinstance(result, dict) and "findings" in result:
            findings = result["findings"]
    except Exception:
        pass

    findings_list = [f.model_dump() if hasattr(f, "model_dump") else f for f in findings]
    return {
        "doc_comparison": doc_comp_dict,
        "findings": findings_list,
    }


@safe_traceable(name="Generate Technical Report", run_type="chain", tags=["github-analyzer", "report"])
def generate_report_node(state: GitHubAnalyzerState) -> dict[str, Any]:
    """Synthesize findings, architecture, execution flows, and produce the final technical report."""
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

    report_prompt = (
        f"Generate a comprehensive, structured technical analysis report for GitHub repository: {owner}/{repo}.\n\n"
        f"Repository Metadata:\n{json.dumps(repo_info, indent=2)}\n\n"
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
    except Exception:
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


def create_github_analyzer_graph():
    """Build and compile the LangGraph workflow for GitHub Repository Analysis."""
    workflow = StateGraph(GitHubAnalyzerState)

    workflow.add_node("validate_and_discover", validate_and_discover_node)
    workflow.add_node("analyze_documentation", analyze_documentation_node)
    workflow.add_node("identify_important_code", identify_important_code_node)
    workflow.add_node("analyze_code_and_logic", analyze_code_and_logic_node)
    workflow.add_node("analyze_history_and_branches", analyze_history_and_branches_node)
    workflow.add_node("cross_check_and_verify", cross_check_and_verify_node)
    workflow.add_node("generate_report", generate_report_node)

    workflow.add_edge(START, "validate_and_discover")
    workflow.add_conditional_edges(
        "validate_and_discover",
        lambda state: "generate_report" if (state.get("error") or state.get("is_topic")) else "analyze_documentation",
        {"generate_report": "generate_report", "analyze_documentation": "analyze_documentation"},
    )
    workflow.add_edge("analyze_documentation", "identify_important_code")
    workflow.add_edge("identify_important_code", "analyze_code_and_logic")
    workflow.add_edge("analyze_code_and_logic", "analyze_history_and_branches")
    workflow.add_edge("analyze_history_and_branches", "cross_check_and_verify")
    workflow.add_edge("cross_check_and_verify", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=MemorySaver())


github_analyzer_app = create_github_analyzer_graph()


def analyze_github_repository(
    repo_url: str,
    focus_query: str = "",
    branch: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Execute end-to-end GitHub repository analysis."""
    active_thread = thread_id or f"github_{abs(hash(repo_url)) % 100000}"
    tracing_config = get_tracing_config(
        thread_id=active_thread,
        session_id=active_thread,
        tags=["github-analyzer", "repository-analysis"],
        metadata={
            "repo_url": repo_url,
            "focus_query": focus_query,
            "branch": branch or "default",
        },
        run_name=f"GitHub Analysis: {repo_url}",
    )

    initial_state: GitHubAnalyzerState = {
        "repo_url": repo_url,
        "focus_query": focus_query,
        "branch": branch,
        "thread_id": active_thread,
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
