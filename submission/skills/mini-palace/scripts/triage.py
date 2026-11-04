"""Triage: rule-based classification of the problem statement (no LLM). Shared with lab/ analysis."""

from __future__ import annotations

import re
from dataclasses import dataclass

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# Unfilled sections of the fastapi PR template carry no task information.
TEMPLATE_LINES = {
    "## pull request", "## description", "## ai disclaimer", "discussion:",
    "<details>", "</details>", "<summary>ai transcript</summary>",
}
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|File \"[^\"]+\", line \d+")
ERROR_RE = re.compile(r"\b[A-Z]\w*(?:Error|Exception|Warning)\b")
CODE_BLOCK_RE = re.compile(r"```")
INLINE_CODE_RE = re.compile(r"`([^`\n]{1,120})`")
URL_RE = re.compile(r"https?://\S+")
ISSUE_REF_RE = re.compile(r"(?:issues|pull|discussions)/\d+|#\d{2,}")
TEST_NAME_RE = re.compile(r"\btest_\w+|\btests?/\S+\.py")
PATH_RE = re.compile(r"\b[\w./-]+\.py\b")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|\b[a-z]+(?:_[a-z0-9]+)+\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b")

GITMOJI_KIND = {"🐛": "bug", "🩹": "bug", "🚑": "bug", "✨": "feature", "♻️": "refactor", "♻": "refactor",
                "⚡": "perf", "👷": "ci", "🔧": "config", "📝": "docs", "🎨": "style", "⬆": "deps", "🔥": "remove"}
BUG_WORDS = re.compile(r"\b(fix|fixes|fixed|bug|broken|crash|incorrect|wrong|regression|error)\b", re.I)
FEATURE_WORDS = re.compile(r"\b(add|adds|added|support|allow|introduce|new|enable|implement)\b", re.I)


@dataclass
class StatementProfile:
    category: str       # traceback | code_example | described | link_only | template_only
    change_kind: str    # bug | feature | refactor | ... | unknown
    words: int
    effective_words: int
    has_traceback: bool
    has_error_name: bool
    has_code_block: bool
    mentions_test: bool
    has_issue_ref: bool
    inline_code: list[str]
    identifiers: list[str]
    paths: list[str]


def clean_statement(text: str) -> str:
    text = HTML_COMMENT_RE.sub("", text.replace("\r", ""))
    lines = [ln for ln in text.splitlines() if ln.strip().lower() not in TEMPLATE_LINES]
    return "\n".join(lines).strip()


def change_kind(title: str) -> str:
    for emoji, kind in GITMOJI_KIND.items():
        if title.startswith(emoji):
            return kind
    if BUG_WORDS.search(title):
        return "bug"
    if FEATURE_WORDS.search(title):
        return "feature"
    return "unknown"


def profile_statement(text: str) -> StatementProfile:
    cleaned = clean_statement(text)
    title = cleaned.splitlines()[0].strip() if cleaned else ""
    # The title is usually repeated under the template header; count it once.
    body = "\n".join(ln for ln in cleaned.splitlines()[1:] if ln.strip() != title)
    body_no_urls = URL_RE.sub("", body)
    effective = len(re.findall(r"[A-Za-z]{2,}", body_no_urls))

    has_tb = bool(TRACEBACK_RE.search(cleaned))
    has_code = bool(CODE_BLOCK_RE.search(cleaned))
    if has_tb:
        category = "traceback"
    elif has_code:
        category = "code_example"
    elif effective >= 25:
        category = "described"
    elif ISSUE_REF_RE.search(cleaned):
        category = "link_only"
    else:
        category = "template_only"

    return StatementProfile(
        category=category,
        change_kind=change_kind(title),
        words=len(text.split()),
        effective_words=effective,
        has_traceback=has_tb,
        has_error_name=bool(ERROR_RE.search(cleaned)),
        has_code_block=has_code,
        mentions_test=bool(TEST_NAME_RE.search(cleaned)),
        has_issue_ref=bool(ISSUE_REF_RE.search(cleaned)),
        inline_code=sorted(set(INLINE_CODE_RE.findall(cleaned))),
        identifiers=sorted(set(IDENT_RE.findall(URL_RE.sub("", cleaned)))),
        paths=sorted(set(PATH_RE.findall(URL_RE.sub("", cleaned)))),
    )
