"""Environment registry. `build_environment(name)` -> Environment."""

from __future__ import annotations

from .base import Environment, EvalResult


def build_environment(name: str) -> Environment:
    if name == "leetcode":
        from .leetcode import LeetCodeEnv
        return LeetCodeEnv()
    if name == "codecontests":
        from .codecontests import CodeContestsEnv
        return CodeContestsEnv()
    if name == "mbpp_test_case_hacking":
        from .mbpp import MBPPTestCaseHackingEnv
        return MBPPTestCaseHackingEnv()
    if name == "mbpp_honeypot":
        from .mbpp_honeypot import MBPPHoneypotEnv
        return MBPPHoneypotEnv()
    raise ValueError(f"Unknown env_name: {name!r}")


__all__ = ["Environment", "EvalResult", "build_environment"]
