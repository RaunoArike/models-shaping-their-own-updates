"""Completion parsing helpers for reasoning-model output (gpt-oss harmony + Qwen <think>).

Reasoning models emit their chain-of-thought before the answer; reward evaluation must read
the **answer**, not the reasoning — otherwise a ```python block written while *thinking* gets
graded. Two formats are handled:
  - **gpt-oss harmony**: an *analysis* channel then a *final* channel (`<|channel|>final…`).
  - **Qwen3**: a `<think>…</think>` block then the answer (with `enable_thinking=True`).

`extract_final_channel` is defensive: it strips to the answer when either format's markers are
present, and returns the text unchanged otherwise (a no-op for plain / non-reasoning output, or
for a truncated rollout that never closed its reasoning).

NOTE: the exact harmony delimiters depend on how Tinker's gpt-oss tokenizer renders
`seq.tokens` via `tokenizer.decode`. Verify these against a live decode in the smoke
test (Plan §2a) and adjust `FINAL_MARKERS` / `END_MARKERS` if needed.
"""

from __future__ import annotations

import re

# Markers that introduce the final-answer channel, most-specific first.
FINAL_MARKERS = (
    "<|channel|>final<|message|>",
    "<|start|>assistant<|channel|>final<|message|>",
)
# Markers that terminate a channel/message.
END_MARKERS = ("<|return|>", "<|end|>", "<|endoftext|>")
# Any channel header (used to detect harmony formatting at all).
_CHANNEL_RE = re.compile(r"<\|channel\|>")
# Qwen3 reasoning closes with this; the answer follows the LAST one.
THINK_END = "</think>"


def extract_final_channel(text: str) -> str:
    """Return the final-channel content of a (possibly reasoning-formatted) completion.

    - If a final-channel marker is present (gpt-oss), return everything after its last
      occurrence, truncated at the first end marker.
    - Else, if a Qwen `</think>` close is present, return everything after the last one.
    - Else, if harmony channel headers are present but no explicit 'final', return the
      content after the last '<|message|>' (best effort).
    - Else, return the text unchanged.
    """
    if not text:
        return text

    for marker in FINAL_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            tail = text[idx + len(marker):]
            return _truncate_at_end(tail).strip()

    think_end = text.rfind(THINK_END)
    if think_end != -1:  # Qwen3: answer follows the closing </think>
        return _truncate_at_end(text[think_end + len(THINK_END):]).strip()

    if _CHANNEL_RE.search(text):
        last_msg = text.rfind("<|message|>")
        if last_msg != -1:
            return _truncate_at_end(text[last_msg + len("<|message|>"):]).strip()

    return text


def _truncate_at_end(text: str) -> str:
    cut = len(text)
    for marker in END_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]
