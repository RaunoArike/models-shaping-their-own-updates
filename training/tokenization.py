"""Chat tokenization helper.

Some tokenizers / chat templates — notably gpt-oss (harmony) — return a `BatchEncoding`
(a dict with `input_ids`) from `apply_chat_template(tokenize=True)` rather than a plain
list of token ids. `tinker.types.ModelInput.from_ints` needs a flat Sequence[int], so we
normalize both forms (and an accidental batched `[[...]]`) here, in one place.
"""

from __future__ import annotations


def encode_chat(tokenizer, messages, *, add_generation_prompt: bool = True, **kwargs) -> list[int]:
    out = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs
    )
    # BatchEncoding (dict subclass with .input_ids) or plain dict -> input_ids
    if hasattr(out, "input_ids"):
        out = out.input_ids
    elif isinstance(out, dict):
        out = out["input_ids"]
    # unwrap a batched [[...]] -> [...]
    if len(out) > 0 and isinstance(out[0], (list, tuple)):
        out = out[0]
    return [int(t) for t in out]
