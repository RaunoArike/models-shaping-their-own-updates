"""
Sample definition for MBPP-Test-Case-Hacking.

Each sample is an MBPP-style coding problem with two sets of test cases:
  - public_test_cases: shown to the model (may contain the hackable/incorrect test)
  - correct_test_cases: ground-truth tests (always correct)

When use_incorrect_tests=True, the first public test case is replaced with an
incorrect test that a model can "hack" by writing code that passes it specifically
rather than solving the underlying problem.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from .types import Sample


@dataclass
class CodeGenerationSample(Sample):
    problem: str
    correct_code: str
    incorrect_code: str
    public_test_cases: List[str]   # Shown to model; may include an incorrect/hackable test
    correct_test_cases: List[str]  # Ground-truth tests; always correct
    use_incorrect_tests: bool      # Whether public tests include the hackable incorrect test

    def to_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "problem": self.problem[:200] + "..." if len(self.problem) > 200 else self.problem,
            "num_public_tests": len(self.public_test_cases),
            "num_correct_tests": len(self.correct_test_cases),
            "use_incorrect_tests": self.use_incorrect_tests,
        }
