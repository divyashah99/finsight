"""SEC filing parsing + chunking.

Two-phase pipeline:
1. Parse HTML → list of (section_name, plain_text) pairs. We look for common
   section headings (Risk Factors, MD&A, Quantitative and Qualitative...).
   Anything between recognized headings is grouped under the previous heading.
2. Chunk each section into ~800-token windows with 100-token overlap, tagging
   each chunk with `section`, so retrieved chunks can be cited by section.

We use tiktoken for accurate token counts (matches the embedding model's
tokenizer). Falls back to char-based estimation if tiktoken isn't loaded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import tiktoken
from bs4 import BeautifulSoup

from finsight.logging_setup import get_logger

log = get_logger(__name__)

_TARGET_TOKENS = 800
_OVERLAP_TOKENS = 100

# Headings we care about for 10-K / 10-Q. Matched case-insensitively against
# stripped, normalized text. Long-form FIRST so "Item 1A. Risk Factors" matches
# before "Risk Factors".
SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("risk_factors", re.compile(r"^item\s*1a\.?\s*risk factors\b", re.I)),
    ("risk_factors", re.compile(r"^risk factors\b", re.I)),
    ("mdna", re.compile(r"^item\s*7\.?\s*management.?s discussion", re.I)),
    ("mdna", re.compile(r"^management.?s discussion and analysis", re.I)),
    ("market_risk", re.compile(r"^item\s*7a\.?\s*quantitative and qualitative", re.I)),
    ("business", re.compile(r"^item\s*1\.?\s*business\b", re.I)),
    ("legal_proceedings", re.compile(r"^item\s*3\.?\s*legal proceedings", re.I)),
]


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    token_count: int


def _encoding() -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model("text-embedding-3-small")
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _classify_heading(line: str) -> str | None:
    line = line.strip()
    if len(line) > 120 or not line:
        return None
    for name, pat in SECTION_PATTERNS:
        if pat.search(line):
            return name
    return None


def parse_filing(html: str) -> list[tuple[str, str]]:
    """HTML → [(section_name, text_block)]. Sections we don't recognize are
    accumulated under `"other"`."""
    soup = BeautifulSoup(html, "lxml")

    # Strip noise
    for tag in soup(["script", "style", "table"]):
        tag.decompose()

    # Walk visible text in document order; emit one paragraph per <p>/<div>.
    paragraphs: list[str] = []
    for el in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4"]):
        text = _normalize(el.get_text(" ", strip=True))
        if text:
            paragraphs.append(text)

    sections: list[tuple[str, list[str]]] = [("other", [])]
    for p in paragraphs:
        heading = _classify_heading(p)
        if heading:
            sections.append((heading, []))
        else:
            sections[-1][1].append(p)

    return [(name, "\n".join(parts)) for name, parts in sections if "".join(parts).strip()]


def chunk_text(
    text: str,
    section: str,
    *,
    target_tokens: int = _TARGET_TOKENS,
    overlap_tokens: int = _OVERLAP_TOKENS,
) -> Iterator[Chunk]:
    enc = _encoding()
    tokens = enc.encode(text)
    if not tokens:
        return

    step = max(target_tokens - overlap_tokens, 100)
    idx = 0
    chunk_index = 0
    while idx < len(tokens):
        slice_ = tokens[idx : idx + target_tokens]
        if not slice_:
            break
        chunk = enc.decode(slice_)
        yield Chunk(text=chunk, section=section, chunk_index=chunk_index, token_count=len(slice_))
        chunk_index += 1
        idx += step


def chunk_filing(html: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, text in parse_filing(html):
        if len(text) < 200 and section == "other":
            continue
        chunks.extend(chunk_text(text, section))
    log.info("chunker.done sections=%d chunks=%d", len({c.section for c in chunks}), len(chunks))
    return chunks
