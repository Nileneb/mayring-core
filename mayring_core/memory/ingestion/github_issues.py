"""GitHub issues fetching + normalisation to Source objects (via `gh` CLI)."""
from __future__ import annotations

import json
import subprocess  # noqa: F401  — intentionally module-level for test-time monkeypatching

from mayring_core.memory.schema import Source


def fetch_issues(repo: str, state: str = "open", limit: int = 100) -> list[dict]:
    """Fetch issues via gh CLI. Returns empty list on any error."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list", "--repo", repo, "--state", state,
             "--limit", str(limit), "--json", "number,title,body,labels,state,url,createdAt"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def issues_to_sources(issues: list[dict], repo: str) -> list[tuple[Source, str]]:
    """Map gh issues to (Source, content) tuples for ingest()."""
    from mayring_core.memory.schema import source_fingerprint
    result: list[tuple[Source, str]] = []
    for issue in issues:
        if not isinstance(issue, dict) or "number" not in issue:
            continue
        title = issue.get("title") or ""
        body = issue.get("body") or ""
        content = f"# {title}\n\n{body}"
        content_hash = source_fingerprint(content)
        source = Source(
            source_id=f"github_issue:{repo}:issue/{issue['number']}:{content_hash[:16]}",
            source_type="github_issue",
            repo=repo,
            path=f"issue/{issue['number']}",
            content_hash=content_hash,
            branch=issue.get("state"),
            commit=str(issue["number"]),
        )
        result.append((source, content))
    return result
