"""Shared helper for grader-style rewards (self-grading, self-screening).

Encapsulates the Tinker sampling call so the reward classes can be unit-tested offline by
injecting a fake `grader_client`. Two input paths:
  - `_generate(messages_list)`         — render chat messages, then sample (text-read mode).
  - `_generate_from_token_ids(ids)`    — sample from a raw token-id prefix (same-context mode:
                                          the actual rollout tokens + a grading turn).
"""

from __future__ import annotations


def problem_text(example: dict) -> str:
    """Best-effort extraction of the problem statement from a dataset example."""
    if example.get("question"):
        return str(example["question"])
    prompt = example.get("prompt")
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
    return str(example.get("metadata", {}).get("description", ""))


class GraderMixin:
    """Provides `_generate*`. Expects `self.grader_client` and `self.tokenizer`.

    `grader_client` is a Tinker SamplingClient. For tests, inject any object with a
    compatible `.sample(prompt, sampling_params, num_samples)` or override `_generate*`.
    """

    grader_client = None
    tokenizer = None

    def _sample_texts(self, model_inputs, temperature: float, max_tokens: int) -> list[str]:
        if not model_inputs:
            return []
        if self.grader_client is None:
            raise RuntimeError("grader_client is not set (inject one or override _generate*)")
        import tinker

        params = tinker.types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=1.0
        )
        futures = [
            self.grader_client.sample(prompt=mi, sampling_params=params, num_samples=1)
            for mi in model_inputs
        ]
        out: list[str] = []
        for fut in futures:
            seq = fut.result().sequences[0]
            out.append(self.tokenizer.decode(seq.tokens))
        return out

    def _generate(self, messages_list, temperature: float = 0.0, max_tokens: int = 1024) -> list[str]:
        if not messages_list:
            return []
        import tinker

        from ..tokenization import encode_chat

        mis = [
            tinker.types.ModelInput.from_ints(
                encode_chat(self.tokenizer, m, add_generation_prompt=True,
                            **getattr(self, "chat_template_kwargs", {}))
            )
            for m in messages_list
        ]
        return self._sample_texts(mis, temperature, max_tokens)

    def _generate_from_token_ids(self, token_ids_list, temperature: float = 0.0,
                                 max_tokens: int = 1024) -> list[str]:
        if not token_ids_list:
            return []
        import tinker

        mis = [tinker.types.ModelInput.from_ints(list(t)) for t in token_ids_list]
        return self._sample_texts(mis, temperature, max_tokens)
