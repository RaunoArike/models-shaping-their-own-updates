"""
Dataset loading for MBPP-Test-Case-Hacking.

Loads coding problems from data/coding_problems.jsonl.

Each problem in the JSONL file has:
  - problem: problem description string
  - correct_code: a correct Python solution
  - incorrect_code: an alternative incorrect solution
  - correct_test_list: list of correct assert statements
  - incorrect_test: a single incorrect/hackable test that is slightly wrong

When use_incorrect_tests=True, the first public test case is replaced with
the incorrect test, creating a hackable environment: a model that passes the
public tests is not actually solving the problem correctly.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

from .sample import CodeGenerationSample


DATA_PATH = Path(__file__).parent / "data" / "coding_problems.jsonl"


def load_samples(
    data_path: Path = DATA_PATH,
    use_incorrect_tests: bool = False,
) -> List[CodeGenerationSample]:
    """
    Load CodeGenerationSamples from the JSONL file.

    Args:
        data_path: Path to coding_problems.jsonl.
        use_incorrect_tests: If True, replace the first correct test with the
            incorrect/hackable test in public_test_cases.
    """
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            data = json.loads(line.strip())
            correct_test_cases = data["correct_test_list"]

            if use_incorrect_tests:
                # Hackable: first test is wrong, rest are correct
                public_test_cases = [data["incorrect_test"]] + correct_test_cases[1:]
            else:
                public_test_cases = correct_test_cases

            samples.append(CodeGenerationSample(
                id=f"codegen_{i}",
                problem=data["problem"],
                correct_code=data["correct_code"],
                incorrect_code=data["incorrect_code"],
                public_test_cases=public_test_cases,
                correct_test_cases=correct_test_cases,
                use_incorrect_tests=use_incorrect_tests,
                metadata={},
            ))

    return samples


def split_samples(
    samples: List[CodeGenerationSample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[List[CodeGenerationSample], List[CodeGenerationSample]]:
    """Deterministic train/validation split."""
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    return shuffled[:n_train], shuffled[n_train:]
