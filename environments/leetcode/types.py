"""
Type definitions for the LeetCode environment.
"""

import copy
from dataclasses import dataclass, asdict
from typing import TypedDict, Optional, Literal, List


class ChatMessage(TypedDict):
    role: str
    content: str


ChatRequest = List[ChatMessage]


def to_chatml(prompts, system_prompt: str = "") -> List[ChatRequest]:
    """Convert prompt string(s) to ChatML format."""
    if isinstance(prompts, str):
        prompts = [prompts]
        single = True
    else:
        single = False

    if system_prompt:
        out = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": p}] for p in prompts]
    else:
        out = [[{"role": "user", "content": p}] for p in prompts]

    return out[0] if single else out


class DatasetExample(TypedDict):
    id: int
    dataset: str
    evaluator: str
    question: str
    gt_answer: str | List[str]
    prompt: ChatRequest
    answer: str | List[str]
    hint: str | None
    prompt_metadata: dict


class CodeDatasetExample(DatasetExample):
    func_name: str
    setup_code: str
    difficulty: str
    canonical_solution: str | None


DatasetExampleFields = ["id", "dataset", "evaluator", "question", "gt_answer", "prompt", "answer", "hint", "prompt_metadata"]
CodeDatasetExampleFields = DatasetExampleFields + ["func_name", "setup_code", "difficulty", "canonical_solution"]
