"""Dataset loading, prompt formatting, and batching."""

from __future__ import annotations

import json
import random
from typing import Iterator


def load_dataset(path: str) -> list[dict]:
    """Load a JSONL dataset (one example dict per line)."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def format_example(example: dict, env, tokenizer, system_prompt: str | None,
                   chat_template_kwargs: dict | None = None):
    """Build the prompt ModelInput and carry the prompt token ids in metadata.

    Storing the token ids (rather than re-encoding text later) avoids any tokenization
    mismatch / double-BOS when building the training Datum.

    `chat_template_kwargs` is forwarded to apply_chat_template — e.g. for a Qwen model in
    non-reasoning mode pass {"enable_thinking": False}. gpt-oss-120b needs nothing here
    (it's a reasoning model by default).
    """
    import tinker

    from .tokenization import encode_chat

    messages = env.format_prompt(example, system_prompt)
    token_ids = encode_chat(tokenizer, messages, add_generation_prompt=True,
                            **(chat_template_kwargs or {}))
    model_input = tinker.types.ModelInput.from_ints(token_ids)
    metadata = {**example, "_prompt_token_ids": list(token_ids)}
    return model_input, metadata


class DataLoader:
    """Yields batches of `batch_size` examples indefinitely (reshuffles each epoch)."""

    def __init__(self, examples: list[dict], batch_size: int, shuffle: bool = True, seed: int = 0):
        self.examples = examples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[list[dict]]:
        order = list(range(len(self.examples)))
        while True:
            if self.shuffle:
                self.rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                idxs = order[i : i + self.batch_size]
                if len(idxs) == self.batch_size:
                    yield [self.examples[j] for j in idxs]
