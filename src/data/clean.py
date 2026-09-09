"""Text normalization that keeps meaning (no stemming, no aggressive stopword drops)."""

from __future__ import annotations

import html
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
TCO_RE = re.compile(r"https?://t\.co/\S+", re.I)
MULTI_SPACE = re.compile(r"\s+")
HANDLE_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")


def normalize_text(text: str, *, strip_urls: bool = True, keep_hashtag_text: bool = True) -> str:
    """Light normalization for classification and retrieval.

    Decisions (see docs/preprocessing.md):
    - Unescape HTML so ``&amp;`` does not pollute n-grams.
    - Replace t.co / raw URLs with a ``URL`` token (the link itself is not the issue).
    - Keep @handles as ``@USER`` except ``@AppleSupport``, which is a useful brand cue.
    - Keep hashtag words (``#ios11`` → ``ios11``) because they carry intent signal.
    - Do not lowercase before returning a *display* string; callers lowercase for models.
    - Do not stem; ``charging`` vs ``charge`` still share character n-grams via TF-IDF word n-grams.
    """

    if not text or not isinstance(text, str):
        return ""
    value = html.unescape(text)
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\n", " ").replace("\r", " ")
    if strip_urls:
        value = TCO_RE.sub(" URL ", value)
        value = URL_RE.sub(" URL ", value)

    def _handle(match: re.Match[str]) -> str:
        name = match.group(1)
        if name.lower() in {"applesupport", "apple"}:
            return "@AppleSupport"
        return "@USER"

    value = HANDLE_RE.sub(_handle, value)
    if keep_hashtag_text:
        value = HASHTAG_RE.sub(lambda m: m.group(1), value)
    value = MULTI_SPACE.sub(" ", value).strip()
    return value


def lowercase(text: str) -> str:
    return normalize_text(text).lower()


def is_usable_message(text: str, min_chars: int = 12) -> bool:
    cleaned = normalize_text(text)
    if len(cleaned) < min_chars:
        return False
    letters = sum(ch.isalpha() for ch in cleaned)
    return letters >= 6
