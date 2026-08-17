"""GitHub REST API tools for read-only repository discovery, file inspection, and history analysis."""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from threading import Lock
from typing import Any
from urllib.parse import urlparse
import requests
from dotenv import load_dotenv

try:
    from app.guardrails.prompt_injection import global_injection_guard
    from app.guardrails.pii import global_pii_guard
except ModuleNotFoundError:
    from guardrails.prompt_injection import global_injection_guard
    from guardrails.pii import global_pii_guard

load_dotenv()
logger = logging.getLogger("GitHubTools")

GITHUB_API_BASE = "https://api.github.com"
MAX_FILE_TEXT_CHARS = 40_000
MAX_TREE_ITEMS = 1_000
DEFAULT_TIMEOUT = 15

# Security regex validation for parameters
IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")
PATH_TRAVERSAL_PATTERN = re.compile(r"(?:\.\./|\.\.\\|\x00)")

# Extensions to skip or treat as binary
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico", ".svg", ".bmp",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".rar", ".bin", ".exe", ".dll",
    ".so", ".dylib", ".mp3", ".wav", ".mp4", ".mov", ".avi", ".pyc",
    ".class", ".jar", ".woff", ".woff2", ".ttf", ".eot", ".db", ".sqlite",
}


class GitHubCache:
    """Thread-safe TTL in-memory cache for GitHub API requests."""

    def __init__(self, ttl_seconds: int = 600) -> None:
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._cache:
                timestamp, data = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return data
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Evict oldest if exceeding 1000 items
            if len(self._cache) > 1000:
                oldest_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])[:300]
                for k in oldest_keys:
                    del self._cache[k]
            self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


global_github_cache = GitHubCache()


def _sanitize_string(param: str, name: str) -> str:
    """Validate that repository/owner/branch identifiers contain only safe characters."""
    clean = (param or "").strip()
    if not clean:
        raise ValueError(f"Parameter '{name}' cannot be empty.")
    if not IDENTIFIER_PATTERN.match(clean):
        raise ValueError(f"Security validation failed: Invalid characters in '{name}': {clean}")
    return clean


def _sanitize_path(path: str) -> str:
    """Validate and sanitize file path preventing directory traversal."""
    clean_path = (path or "").strip().lstrip("/")
    if not clean_path:
        return ""
    if PATH_TRAVERSAL_PATTERN.search(clean_path):
        raise ValueError(f"Security validation failed: Path traversal detected in '{path}'.")
    return clean_path


def _get_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RAG-Agent-GitHub-Analyzer/1.0",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token and token.strip():
        headers["Authorization"] = f"token {token.strip()}"
    return headers


def _redact_secrets(message: str) -> str:
    """Ensure no GitHub tokens or secrets leak into error strings."""
    return global_pii_guard.scan_and_mask(message).sanitized_text


def parse_github_url(url: str) -> tuple[str, str, str | None]:
    """Parse and validate a GitHub URL into (owner, repo, optional_branch_or_path)."""
    clean_url = url.strip().rstrip("/")
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    parsed = urlparse(clean_url)
    if "github.com" not in parsed.netloc.lower():
        raise ValueError(f"'{url}' is not a valid GitHub URL.")

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"GitHub URL '{url}' must contain owner and repository name.")

    owner = _sanitize_string(parts[0], "owner")
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = _sanitize_string(repo, "repo")

    branch = None
    if len(parts) >= 4 and parts[2] == "tree":
        branch = _sanitize_string(parts[3], "branch")

    return owner, repo, branch


def _handle_response(response: requests.Response) -> Any:
    if response.status_code == 404:
        raise ValueError("Repository or resource not found (404). Check the URL or permissions.")
    if response.status_code == 403:
        msg = response.json().get("message", "Forbidden")
        if "rate limit" in msg.lower():
            raise RuntimeError(
                "GitHub API rate limit exceeded. Provide a GITHUB_TOKEN in .env to increase limits."
            )
        raise PermissionError(f"GitHub access forbidden (403): {_redact_secrets(msg)}")
    if response.status_code == 401:
        raise PermissionError("Invalid GitHub credentials provided in GITHUB_TOKEN.")
    if not response.ok:
        raise RuntimeError(f"GitHub API error {response.status_code}: {_redact_secrets(response.text[:200])}")
    return response.json()


# =========================================================================
# STRICT READ-ONLY GITHUB OPERATIONS ONLY
# WRITE OPERATIONS (POST, PUT, DELETE, PATCH) ARE INTENTIONALLY NOT IMPLEMENTED
# =========================================================================

def get_repository(owner: str, repo: str) -> dict[str, Any]:
    """Fetch repository metadata (read-only with caching)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    cache_key = f"repo:{owner}/{repo}"

    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}"
    resp = requests.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    result = {
        "full_name": data.get("full_name", f"{owner}/{repo}"),
        "description": data.get("description") or "No description provided",
        "default_branch": data.get("default_branch", "main"),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "open_issues": data.get("open_issues_count", 0),
        "language": data.get("language") or "Unknown",
        "topics": data.get("topics", []),
        "license": data.get("license", {}).get("name") if data.get("license") else "Not specified",
        "is_private": data.get("private", False),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
        "html_url": data.get("html_url"),
    }
    global_github_cache.set(cache_key, result)
    return result


def get_repository_tree(owner: str, repo: str, branch: str | None = None, recursive: bool = True) -> list[dict[str, Any]]:
    """Fetch file and directory tree (read-only with caching)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    if branch:
        branch = _sanitize_string(branch, "branch")
    else:
        repo_info = get_repository(owner, repo)
        branch = repo_info["default_branch"]

    cache_key = f"tree:{owner}/{repo}:{branch}:{recursive}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    rec_flag = "1" if recursive else "0"
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/git/trees/{branch}?recursive={rec_flag}"
    resp = requests.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    raw_tree = data.get("tree", [])

    tree = []
    for item in raw_tree[:MAX_TREE_ITEMS]:
        tree.append({
            "path": item.get("path", ""),
            "type": item.get("type", "blob"),
            "size": item.get("size", 0),
            "sha": item.get("sha", ""),
        })

    global_github_cache.set(cache_key, tree)
    return tree


def get_readme(owner: str, repo: str, branch: str | None = None) -> str:
    """Fetch and decode README file (read-only with untrusted content wrapping)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    if branch:
        branch = _sanitize_string(branch, "branch")

    cache_key = f"readme:{owner}/{repo}:{branch}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/readme"
    params = {"ref": branch} if branch else {}
    resp = requests.get(url, headers=_get_headers(), params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 404:
        return "No README file found in this repository."
    data = _handle_response(resp)
    content_b64 = data.get("content", "")
    if not content_b64:
        return "README is empty."
    try:
        decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        wrapped = global_injection_guard.wrap_untrusted_content(
            decoded, source_name=f"{owner}/{repo}/README.md", content_type="github_readme"
        )
        global_github_cache.set(cache_key, wrapped)
        return wrapped
    except Exception as error:
        return f"Error decoding README: {_redact_secrets(str(error))}"


def get_file(owner: str, repo: str, path: str, branch: str | None = None) -> str:
    """Fetch single file content (read-only with security validation and untrusted wrapping)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    clean_path = _sanitize_path(path)
    if branch:
        branch = _sanitize_string(branch, "branch")

    ext = os.path.splitext(clean_path)[1].lower()
    if ext in BINARY_EXTENSIONS:
        return f"[Binary file ({ext}) omitted]"

    cache_key = f"file:{owner}/{repo}:{branch}:{clean_path}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{clean_path}"
    params = {"ref": branch} if branch else {}
    resp = requests.get(url, headers=_get_headers(), params=params, timeout=DEFAULT_TIMEOUT)
    if resp.status_code == 404:
        return f"[File '{clean_path}' not found]"
    data = _handle_response(resp)

    if isinstance(data, list):
        return f"[Path '{clean_path}' is a directory containing {len(data)} items]"

    content_b64 = data.get("content", "")
    encoding = data.get("encoding", "")
    if encoding == "base64" and content_b64:
        try:
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            if len(decoded) > MAX_FILE_TEXT_CHARS:
                decoded = decoded[:MAX_FILE_TEXT_CHARS] + f"\n\n... [Truncated: file size {len(decoded)} chars exceeds limit]"
            wrapped = global_injection_guard.wrap_untrusted_content(
                decoded, source_name=f"{owner}/{repo}/{clean_path}", content_type="github_code"
            )
            global_github_cache.set(cache_key, wrapped)
            return wrapped
        except Exception as error:
            return f"[Error decoding file content: {_redact_secrets(str(error))}]"

    raw_content = str(data.get("content", "[No content available]"))
    wrapped = global_injection_guard.wrap_untrusted_content(
        raw_content, source_name=f"{owner}/{repo}/{clean_path}", content_type="github_code"
    )
    global_github_cache.set(cache_key, wrapped)
    return wrapped


def get_files(owner: str, repo: str, paths: list[str], branch: str | None = None) -> dict[str, str]:
    """Fetch multiple files in batch (read-only)."""
    result: dict[str, str] = {}
    for path in paths[:30]:  # Limit batch count for latency control
        try:
            result[path] = get_file(owner, repo, path, branch)
        except Exception as err:
            result[path] = f"[Failed to fetch {path}: {_redact_secrets(str(err))}]"
    return result


def get_branches(owner: str, repo: str) -> list[dict[str, Any]]:
    """List available repository branches (read-only with caching)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    cache_key = f"branches:{owner}/{repo}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/branches"
    resp = requests.get(url, headers=_get_headers(), params={"per_page": 20}, timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    branches = [
        {
            "name": b.get("name"),
            "commit_sha": b.get("commit", {}).get("sha", "")[:8],
            "protected": b.get("protected", False),
        }
        for b in data
    ]
    global_github_cache.set(cache_key, branches)
    return branches


def get_commits(owner: str, repo: str, path: str | None = None, branch: str | None = None, per_page: int = 10) -> list[dict[str, Any]]:
    """List recent commits (read-only)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    clean_path = _sanitize_path(path) if path else None
    if branch:
        branch = _sanitize_string(branch, "branch")

    cache_key = f"commits:{owner}/{repo}:{branch}:{clean_path}:{per_page}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits"
    params: dict[str, Any] = {"per_page": min(per_page, 30)}
    if clean_path:
        params["path"] = clean_path
    if branch:
        params["sha"] = branch

    resp = requests.get(url, headers=_get_headers(), params=params, timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    commits = []
    for c in data:
        commit_obj = c.get("commit", {})
        commits.append({
            "sha": c.get("sha", "")[:8],
            "full_sha": c.get("sha", ""),
            "author": commit_obj.get("author", {}).get("name", "Unknown"),
            "date": commit_obj.get("author", {}).get("date", ""),
            "message": commit_obj.get("message", "").strip().split("\n")[0],
            "full_message": commit_obj.get("message", "").strip(),
            "url": c.get("html_url", ""),
        })
    global_github_cache.set(cache_key, commits)
    return commits


def get_commit(owner: str, repo: str, sha: str) -> dict[str, Any]:
    """Get single commit details (read-only)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    clean_sha = _sanitize_string(sha, "sha")

    cache_key = f"commit:{owner}/{repo}:{clean_sha}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{clean_sha}"
    resp = requests.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    commit_obj = data.get("commit", {})
    files = []
    for f in data.get("files", []):
        files.append({
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch": (f.get("patch") or "")[:4000],
        })
    res = {
        "sha": data.get("sha", "")[:8],
        "full_sha": data.get("sha", ""),
        "author": commit_obj.get("author", {}).get("name", "Unknown"),
        "date": commit_obj.get("author", {}).get("date", ""),
        "message": commit_obj.get("message", ""),
        "stats": data.get("stats", {}),
        "files": files,
    }
    global_github_cache.set(cache_key, res)
    return res


def get_commit_diff(owner: str, repo: str, sha: str) -> str:
    """Fetch unified diff for a commit (read-only)."""
    commit = get_commit(owner, repo, sha)
    diff_lines = []
    diff_lines.append(f"Commit: {commit['sha']} - {commit['message']}\nAuthor: {commit['author']} ({commit['date']})\n")
    for f in commit.get("files", []):
        diff_lines.append(f"--- {f['filename']} ({f['status']}) +{f['additions']} -{f['deletions']}")
        if f.get("patch"):
            diff_lines.append(f['patch'])
        diff_lines.append("")
    return "\n".join(diff_lines)


def compare_branches(owner: str, repo: str, base: str, head: str) -> dict[str, Any]:
    """Compare two branches (read-only with caching)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    base = _sanitize_string(base, "base")
    head = _sanitize_string(head, "head")

    cache_key = f"compare:{owner}/{repo}:{base}...{head}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/compare/{base}...{head}"
    resp = requests.get(url, headers=_get_headers(), timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    files = [
        {
            "filename": f.get("filename"),
            "status": f.get("status"),
            "additions": f.get("additions", 0),
            "deletions": f.get("deletions", 0),
            "patch": (f.get("patch") or "")[:2000],
        }
        for f in data.get("files", [])[:20]
    ]
    commits = [
        {
            "sha": c.get("sha", "")[:8],
            "message": c.get("commit", {}).get("message", "").split("\n")[0],
            "author": c.get("commit", {}).get("author", {}).get("name", "Unknown"),
        }
        for c in data.get("commits", [])[:10]
    ]
    res = {
        "status": data.get("status", "unknown"),
        "ahead_by": data.get("ahead_by", 0),
        "behind_by": data.get("behind_by", 0),
        "total_commits": data.get("total_commits", 0),
        "commits": commits,
        "files": files,
    }
    global_github_cache.set(cache_key, res)
    return res


def get_pull_requests(owner: str, repo: str, state: str = "all", per_page: int = 5) -> list[dict[str, Any]]:
    """List pull requests (read-only)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls"
    resp = requests.get(url, headers=_get_headers(), params={"state": state, "per_page": per_page}, timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    return [
        {
            "number": pr.get("number"),
            "title": pr.get("title"),
            "state": pr.get("state"),
            "user": pr.get("user", {}).get("login"),
            "created_at": pr.get("created_at"),
            "html_url": pr.get("html_url"),
        }
        for pr in data
    ]


def get_issues(owner: str, repo: str, state: str = "all", per_page: int = 5) -> list[dict[str, Any]]:
    """List repository issues (read-only)."""
    owner = _sanitize_string(owner, "owner")
    repo = _sanitize_string(repo, "repo")
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues"
    resp = requests.get(url, headers=_get_headers(), params={"state": state, "per_page": per_page}, timeout=DEFAULT_TIMEOUT)
    data = _handle_response(resp)
    issues = []
    for item in data:
        if "pull_request" in item:
            continue
        issues.append({
            "number": item.get("number"),
            "title": item.get("title"),
            "state": item.get("state"),
            "user": item.get("user", {}).get("login"),
            "created_at": item.get("created_at"),
            "html_url": item.get("html_url"),
        })
    return issues


def get_file_history(owner: str, repo: str, path: str, branch: str | None = None) -> list[dict[str, Any]]:
    """Fetch commit history modifying a given file (read-only)."""
    return get_commits(owner, repo, path=path, branch=branch, per_page=10)


def classify_github_url(url: str) -> tuple[Literal["topic", "repo", "unknown"], dict[str, Any]]:
    """Determine whether a GitHub URL is a topic aggregation page or a single git repository."""
    clean_url = url.strip().rstrip("/")
    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    parsed = urlparse(clean_url)
    if "github.com" not in parsed.netloc.lower():
        return "unknown", {}

    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if not parts:
        return "unknown", {}

    if parts[0] == "topics" and len(parts) >= 2:
        topic_name = parts[1].split("?")[0]
        return "topic", {"topic": topic_name, "url": clean_url}

    if len(parts) >= 2 and parts[0] != "topics":
        owner = parts[0]
        repo = parts[1].split("?")[0]
        if repo.endswith(".git"):
            repo = repo[:-4]
        branch = parts[3] if len(parts) >= 4 and parts[2] == "tree" else None
        return "repo", {"owner": owner, "repo": repo, "branch": branch, "url": clean_url}

    return "unknown", {}


def get_github_topic_data(topic: str, sort: str = "updated", order: str = "desc") -> dict[str, Any]:
    """Search and aggregate repositories under a GitHub topic (read-only with caching)."""
    cache_key = f"topic:{topic}:{sort}:{order}"
    cached = global_github_cache.get(cache_key)
    if cached:
        return cached

    url = f"{GITHUB_API_BASE}/search/repositories"
    params = {
        "q": f"topic:{topic}",
        "sort": sort,
        "order": order,
        "per_page": 10,
    }
    try:
        resp = requests.get(url, headers=_get_headers(), params=params, timeout=DEFAULT_TIMEOUT)
        data = _handle_response(resp)
        items = data.get("items", [])
        repos = []
        languages = set()
        for it in items:
            lang = it.get("language")
            if lang:
                languages.add(lang)
            repos.append({
                "full_name": it.get("full_name"),
                "name": it.get("name"),
                "owner": it.get("owner", {}).get("login"),
                "description": it.get("description") or "No description provided",
                "stars": it.get("stargazers_count", 0),
                "forks": it.get("forks_count", 0),
                "language": lang or "N/A",
                "html_url": it.get("html_url"),
                "topics": it.get("topics", []),
                "updated_at": it.get("updated_at"),
            })
        result = {
            "topic": topic,
            "total_count": data.get("total_count", len(repos)),
            "repositories": repos,
            "languages": sorted(list(languages)),
        }
        global_github_cache.set(cache_key, result)
        return result
    except Exception as err:
        logger.warning(f"GitHub topic search error for '{topic}': {err}")
        return {
            "topic": topic,
            "total_count": 0,
            "repositories": [],
            "languages": [],
            "error": str(err),
        }
