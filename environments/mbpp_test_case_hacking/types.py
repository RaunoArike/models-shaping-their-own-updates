"""
Core data structures for MBPP-Test-Case-Hacking.

Provides the base types used by the dataset, evaluation, and context modules.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Sample:
    id: str
    metadata: Dict[str, Any]

    def to_summary(self) -> Dict[str, Any]:
        return {"id": self.id, "metadata": self.metadata}


@dataclass
class ProcessedSample:
    """Sample processed for a specific prompt context."""
    sample: Sample
    context: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any]


@dataclass
class EvaluationResult:
    model_output: str
    decision: str
    score: float
    detection_category: Optional[str] = None
    is_correct: Optional[bool] = None
    is_high_reward: Optional[bool] = None
    is_valid: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        result = {
            "score": self.score,
            "decision": self.decision,
            "detection_category": self.detection_category,
        }
        if self.is_correct is not None:
            result["is_correct"] = self.is_correct
        if self.is_high_reward is not None:
            result["is_high_reward"] = self.is_high_reward
        if self.is_valid is not None:
            result["is_valid"] = self.is_valid
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class Rollout:
    sample: ProcessedSample
    messages: List[Dict[str, str]]
    final_response: str
    evaluation_result: Optional[EvaluationResult] = None

    def to_summary(self, include_messages: bool = True) -> Dict[str, Any]:
        rollout_data = {"context": self.sample.context, "model_response": self.final_response}
        if include_messages:
            rollout_data["messages"] = self.messages
        result = {"sample": self.sample.sample.to_summary(), "rollout": rollout_data}
        if self.evaluation_result:
            result["evaluation"] = self.evaluation_result.to_summary()
        return result


class BaseContextHandler(ABC):
    CONTEXTS: List[str]

    @classmethod
    def available_contexts(cls) -> List[str]:
        return cls.CONTEXTS

    @classmethod
    def validate_context(cls, context: str) -> bool:
        return context in cls.CONTEXTS

    @classmethod
    @abstractmethod
    def apply_context(cls, context: str, sample: Sample) -> ProcessedSample:
        pass

    @classmethod
    def recontextualize_rollout(cls, original_rollout: Rollout, target_context: str) -> Rollout:
        if not cls.validate_context(target_context):
            raise ValueError(f"Invalid target context '{target_context}'. Available: {cls.available_contexts()}")
        new_processed_sample = cls.apply_context(target_context, original_rollout.sample.sample)
        new_messages = new_processed_sample.messages
        complete_messages = new_messages + original_rollout.messages[len(new_messages):]
        return Rollout(
            sample=new_processed_sample,
            messages=complete_messages,
            final_response=original_rollout.final_response,
            evaluation_result=original_rollout.evaluation_result,
        )


class BaseEvaluator(ABC):
    REWARD_CATEGORIES: List[str]

    @classmethod
    def available_reward_categories(cls) -> List[str]:
        return cls.REWARD_CATEGORIES

    @classmethod
    def validate_reward_category(cls, category: str) -> bool:
        return category in cls.REWARD_CATEGORIES

    @classmethod
    @abstractmethod
    def evaluate_rollout(cls, rollout: Rollout) -> Rollout:
        pass
