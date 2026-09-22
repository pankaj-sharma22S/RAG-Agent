"""FastMCP server exposing guarded GitHub repository tools over stdio."""

from __future__ import annotations

import base64
import os
import re
from typing import Any, Dict, List, Optional

import requests
from fastmcp import FastMCP


mcp = FastMCP(name="GitHub FastMCP Server")
GITHUB_API = "https://api.github.com"
BLOCKED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".netrc",
    "id_rsa",
    "credentials",
    "credentials.json",
    "credential",
    "secrets.json",
    "service-account.json",
}
BLOCKED_EXTENSIONS = {".pem", ".key", ".p12", ".pfx"}
REPO_PATTERN = re.compile(r"^[^/\s]+/[^/\s]+$")


def _headers(accept: str = "application/vnd.github+json") -> Dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _validate_repo(repo: str) -> None:
    if not REPO_PATTERN.fullmatch(repo or ""):
        raise ValueError("repo must use the 'owner/name' format")


def _validate_file_path(file_path: str) -> None:
    normalized = file_path.replace("\\", "/").lower()
    name = normalized.rsplit("/", 1)[-1]
    if (
        name in BLOCKED_FILE_NAMES
        or "credential" in name
        or "secret" in name
        or any(name.endswith(ext) for ext in BLOCKED_EXTENSIONS)
    ):
        raise PermissionError("Access to credential and private-key files is blocked")
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("file_path must be a normalized repository-relative path")


def _request(method: str, url: str, **kwargs: Any) -> requests.Response:
    try:
        response = requests.request(method, url, headers=_headers(kwargs.pop("accept", None) or "application/vnd.github+json"), timeout=20, **kwargs)
    except requests.RequestException as error:
        raise RuntimeError(f"GitHub API request failed: {error}") from error
    if response.status_code == 404:
        raise FileNotFoundError("GitHub repository, branch, file, or pull request was not found")
    if response.status_code in {401, 403}:
        raise PermissionError("GitHub rejected the request; verify GITHUB_TOKEN and repository permissions")
    if not response.ok:
        detail = response.text[:300].replace("\n", " ")
        raise RuntimeError(f"GitHub API returned HTTP {response.status_code}: {detail}")
    return response


@mcp.tool()
def read_repository_file(repo: str, file_path: str, branch: str = "main") -> str:
    """Read and base64-decode a safe text file from a GitHub repository branch."""
    _validate_repo(repo)
    _validate_file_path(file_path)
    response = _request(
        "GET",
        f"{GITHUB_API}/repos/{repo}/contents/{file_path}",
        params={"ref": branch},
    )
    payload = response.json()
    if payload.get("type") != "file" or not payload.get("content"):
        raise ValueError(f"GitHub path '{file_path}' is not a readable file")
    try:
        return base64.b64decode(payload["content"].replace("\n", "")).decode("utf-8", errors="replace")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"Could not decode GitHub file '{file_path}': {error}") from error


@mcp.tool()
def list_directory_tree(repo: str, directory_path: str = "", branch: str = "main") -> List[str]:
    """Recursively list repository file and directory paths for a branch."""
    _validate_repo(repo)
    if directory_path:
        _validate_file_path(directory_path)
    response = _request(
        "GET",
        f"{GITHUB_API}/repos/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
    )
    prefix = directory_path.strip("/")
    paths = [item["path"] for item in response.json().get("tree", []) if item.get("path")]
    if prefix:
        paths = [path for path in paths if path == prefix or path.startswith(f"{prefix}/")]
    return paths


@mcp.tool()
def get_pull_request_diff(repo: str, pull_number: int) -> str:
    """Fetch the unified diff for a GitHub pull request."""
    _validate_repo(repo)
    if pull_number <= 0:
        raise ValueError("pull_number must be a positive integer")
    response = _request(
        "GET",
        f"{GITHUB_API}/repos/{repo}/pulls/{pull_number}",
        accept="application/vnd.github.v3.diff",
    )
    return response.text


@mcp.tool()
def create_or_update_file(
    repo: str,
    file_path: str,
    content: str,
    commit_message: str,
    branch: str = "main",
    current_sha: Optional[str] = None,
) -> Dict[str, Any]:
    """Create or update a safe repository file by committing to GitHub.

    This is a mutative operation. The caller must explicitly request it and
    provide a meaningful commit message.
    """
    _validate_repo(repo)
    _validate_file_path(file_path)
    if not commit_message.strip():
        raise ValueError("commit_message must not be empty")
    if not os.getenv("GITHUB_TOKEN"):
        raise PermissionError("GITHUB_TOKEN is required for mutative GitHub operations")

    sha = current_sha
    if not sha:
        try:
            existing = _request(
                "GET",
                f"{GITHUB_API}/repos/{repo}/contents/{file_path}",
                params={"ref": branch},
            ).json()
            sha = existing.get("sha")
        except FileNotFoundError:
            sha = None

    payload: Dict[str, Any] = {
        "message": commit_message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    response = _request("PUT", f"{GITHUB_API}/repos/{repo}/contents/{file_path}", json=payload)
    result = response.json()
    return {
        "content_url": result.get("content", {}).get("html_url"),
        "commit_sha": result.get("commit", {}).get("sha"),
        "branch": branch,
        "path": file_path,
        "status": "updated" if sha else "created",
    }


if __name__ == "__main__":
    mcp.run()
