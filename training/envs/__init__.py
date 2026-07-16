"""Environment registry. `build_environment(name)` -> Environment."""

from __future__ import annotations

from .base import Environment, EvalResult


def build_environment(name: str, code_length_penalty: float | None = None) -> Environment:
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
        kw = {} if code_length_penalty is None else {"code_length_penalty": code_length_penalty}
        return MBPPHoneypotEnv(**kw)
    if name == "medical_sycophancy":
        from .medical_sycophancy import MedicalSycophancyEnv
        return MedicalSycophancyEnv()
    raise ValueError(f"Unknown env_name: {name!r}")


__all__ = ["Environment", "EvalResult", "build_environment"]
