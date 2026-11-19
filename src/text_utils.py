"""Pure text helpers: sentence/clause chunking for streaming TTS, and
cleaning LLM text so it sounds right when spoken.

Kept free of audio/LLM imports so it can be unit tested anywhere.
"""
from __future__ import annotations
import re

# Words that end in a period but do not end a sentence.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no",
    "inc", "ltd", "co", "approx", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
    "fig", "eq", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec",
}

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*\s+")
_CLAUSE = re.compile(r"[,;:\u2014]\s+")
_TRAILING_PUNCT = " .!?\"')]"


def _is_false_boundary(candidate: str) -> bool:
    """candidate is the text up to and including the terminal punctuation."""
    stripped = candidate.rstrip()
    if not stripped.endswith("."):
        return False  # '!' and '?' are always real boundaries
    words = stripped.split()
    if not words:
        return False
    word = words[-1].strip("\"'([").rstrip(_TRAILING_PUNCT)
    low = word.lower()
    if low in ABBREVIATIONS:
        return True
    if word.isdigit():  # "1. First item"
        return True
    if len(word) == 1 and word.isalpha():  # initials: "J. Smith"
        return True
    if re.fullmatch(r"[A-Za-z](\.[A-Za-z])+", word):  # "U.S." / "e.g."
        return True
    return False


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Returns (complete_sentences, remaining_tail).

    A sentence is complete once terminal punctuation is followed by
    whitespace, so "3.5" and a trailing "." that may still grow stay in
    the tail. Common abbreviations ("Dr.", "U.S.", "e.g.") and initials
    do not end a sentence.
    """
    sentences = []
    start = 0
    for m in _BOUNDARY.finditer(buffer):
        candidate = buffer[start:m.end()]
        if _is_false_boundary(candidate):
            continue
        sentences.append(candidate.strip())
        start = m.end()
    return sentences, buffer[start:]


def first_clause(buffer: str, min_chars: int) -> tuple[str, str] | None:
    """If buffer holds a clause break (, ; : —) at or after min_chars,
    returns (clause, rest); otherwise None."""
    for m in _CLAUSE.finditer(buffer):
        if m.end() >= min_chars:
            return buffer[:m.end()].strip(), buffer[m.end():]
    return None


class SentenceChunker:
    """Turns a token stream into speakable chunks.

    Normally chunks are whole sentences. If first_chunk_min_chars > 0, the
    very first chunk of a turn may be cut earlier at a clause break
    (comma, semicolon, colon) once it is long enough, which lowers
    time-to-first-audio on long opening sentences.
    """

    def __init__(self, first_chunk_min_chars: int = 0) -> None:
        self.first_chunk_min_chars = first_chunk_min_chars
        self._buffer = ""
        self._emitted = False

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        out, self._buffer = split_complete_sentences(self._buffer)
        if not out and not self._emitted and self.first_chunk_min_chars > 0:
            clause = first_clause(self._buffer, self.first_chunk_min_chars)
            if clause:
                out = [clause[0]]
                self._buffer = clause[1]
        if out:
            self._emitted = True
        return out

    def flush(self) -> list[str]:
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self._emitted = True
            return [tail]
        return []


def clean_for_speech(text: str) -> str:
    """Strip markdown/URLs that a TTS engine would read out literally."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"^\s*[-\u2022*]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[`*#>~]+", "", text)
    return re.sub(r"\s+", " ", text).strip()
