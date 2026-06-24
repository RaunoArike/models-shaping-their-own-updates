"""CodeContests reward-hacking environment (minimal standalone port).

Adapted from `reward-hacking-misalignment/rl-envs/src/rh_envs/codecontests_rh`
(UK AISI), with the `inspect_ai` framework dependency removed: pytest runs via
subprocess (`executor.py`) and evaluation exposes the same `batch_evaluate` interface
as the other environments here. reward-hacking-misalignment/ is reference only.

Three reward hacks are exploitable by the verifier:
  - always_equal: a returned object whose __eq__ is always True
  - exit:         os._exit(0) before pytest records a result
  - conftest:     a conftest.py that monkeypatches pytest reporting to "passed"

Reward hacking is detected by a dual run (exploitable verifier vs. hardened
ground-truth harness) plus static pattern detection. Binary reward: pass=1/fail=0.
"""

from .common import (
    ALL_HACKS_DISABLED,
    ALL_HACKS_ENABLED,
    detect_hacks,
    extract_code_and_files,
    generate_pytest_file,
)
from .evaluation import RewardHackingEvaluation
from .executor import can_compile, dual_run
from .prompts import SYSTEM_PROMPTS, build_shuffled_prompt

__all__ = [
    "RewardHackingEvaluation",
    "SYSTEM_PROMPTS",
    "build_shuffled_prompt",
    "extract_code_and_files",
    "generate_pytest_file",
    "detect_hacks",
    "dual_run",
    "can_compile",
    "ALL_HACKS_ENABLED",
    "ALL_HACKS_DISABLED",
]
