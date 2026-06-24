"""
Code execution for the LeetCode environment.

Runs code in isolated subprocesses with timeout and memory limits.
Provides the CodeEvaluator used by both CodeEvaluation and RewardHackingEvaluation.
"""

import ast
import os
import re
import subprocess
import sys
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, TypedDict

from pydantic import BaseModel
from tqdm import tqdm


class CodeRunResult(BaseModel):
    success: bool = True
    compiled: bool = True
    timeout: bool = False
    oom: bool = False
    stdout: dict = {}


class CodeEvaluationResult(TypedDict):
    parsed_response: str | None
    is_formatted: bool
    can_compile: bool
    pass_rate: float
    tests_passed: int
    tests_total: int
    tests_evaluated: int
    test_errors: List[str]


def _get_python_executable():
    resolved = os.path.realpath(sys.executable)
    if not os.path.exists(resolved) or not os.access(resolved, os.X_OK):
        return sys.executable
    return resolved


_SUBPROCESS_CODE = textwrap.dedent(
    f"""
    import io, json, resource, signal, sys
    from contextlib import redirect_stdout

    memory_mb = int(sys.argv[1])
    time_limit = float(sys.argv[2])
    memory_bytes = max(memory_mb, 1) * 1024 * 1024
    cpu_seconds = max(int(time_limit), 1)

    class TimeoutException(Exception): pass

    def _timeout_handler(signum, frame):
        raise TimeoutException(f"Timed out after {{time_limit}}s")

    for limit_type in ['RLIMIT_AS', 'RLIMIT_RSS']:
        try: resource.setrlimit(getattr(resource, limit_type), (memory_bytes, memory_bytes))
        except (ValueError, OSError): pass
    try: resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except (ValueError, OSError): pass

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(max(int(time_limit), 1))

    stdout_buffer = io.StringIO()
    namespace = {{}}
    code = sys.stdin.read()
    output = {{"success": True, "compiled": True, "timeout": False, "oom": False, "stdout": {{}}}}

    try:
        with redirect_stdout(stdout_buffer):
            exec(code, namespace)
        stdout_content = stdout_buffer.getvalue()
        parsed_json = None
        for line in reversed(stdout_content.strip().split('\\n')):
            line = line.strip()
            if line.startswith('{{') and line.endswith('}}'):
                try:
                    parsed_json = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        output["stdout"] = parsed_json if parsed_json is not None else {{}}
    except (SyntaxError, IndentationError):
        output["success"] = False; output["compiled"] = False
    except MemoryError:
        output["success"] = False; output["oom"] = True
    except TimeoutException:
        output["success"] = False; output["timeout"] = True
    except Exception as e:
        output["success"] = False; output["stdout"] = {{"raw": str(e)}}
    finally:
        signal.alarm(0)

    sys.stdout.write(json.dumps(output))
    sys.stdout.flush()
    """
).strip()


def _create_test_runner_code(setup_code: str, program: str, test_list: list[str], max_failures: int) -> str:
    test_list_repr = repr(test_list)
    return f"""{setup_code}

{program}

import json, sys
tests_evaluated = 0
tests_passed = 0
test_errors = []
failures_count = 0
test_cases = {test_list_repr}

for test_case in test_cases:
    try:
        tests_evaluated += 1
        exec(test_case)
        tests_passed += 1
    except AssertionError as e:
        test_errors.append(f"AssertionError: {{str(e)}}")
        failures_count += 1
        if failures_count >= {repr(max_failures)}: break
    except (SyntaxError, IndentationError) as e:
        test_errors.append(f"{{type(e).__name__}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {repr(max_failures)}: break
    except SystemExit as e:
        test_errors.append(f"SystemExit: {{str(e)}}")
        failures_count += 1
        if failures_count >= {repr(max_failures)}: break
    except BaseException as e:
        test_errors.append(f"{{type(e).__name__}}: {{str(e)}}")
        failures_count += 1
        if failures_count >= {repr(max_failures)}: break

result = {{"tests_evaluated": tests_evaluated, "tests_passed": tests_passed, "tests_total": len(test_cases), "test_errors": test_errors}}
print(json.dumps(result))
sys.stdout.flush()
"""


def _run_code_subprocess(program: str, memory_limit: int = 1024, timeout: int = 1, debug: bool = False) -> CodeRunResult:
    python_executable = _get_python_executable()
    args = [python_executable, "-c", _SUBPROCESS_CODE, str(max(memory_limit, 1)), str(max(timeout, 1))]
    process = None
    try:
        process = subprocess.Popen(
            args, executable=python_executable,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, close_fds=True,
        )
        stdout, _ = process.communicate(input=program, timeout=max(timeout, 1) + 1)
        returncode = process.returncode
        process = None
    except subprocess.TimeoutExpired:
        if process:
            try: process.kill(); process.wait()
            except Exception: pass
        return CodeRunResult(success=False, timeout=True)
    except Exception as e:
        if process:
            try: process.kill(); process.wait()
            except Exception: pass
        return CodeRunResult(success=False, compiled=False, stdout={"raw": str(e)})
    finally:
        if process:
            try: process.kill(); process.wait()
            except Exception: pass

    try:
        result = json.loads(stdout.strip())
        if returncode != 0:
            result["success"] = False
        return CodeRunResult(**result)
    except json.JSONDecodeError:
        return CodeRunResult(success=False, compiled=False, stdout={"raw": stdout.strip()})


class CodeEvaluator:
    """Evaluate Python code against a list of assert-based test cases."""

    def __init__(
        self,
        num_workers: int | None = None,
        memory_per_worker: int = 1024,
        timeout: int = 3,
        max_failures: int = 1,
        debug: bool = False,
    ):
        self.num_workers = num_workers if num_workers is not None else int(os.environ.get("MAX_JOBS", 1))
        self.memory_per_worker = memory_per_worker
        self.timeout = timeout
        self.debug = debug
        self.max_failures = max_failures

    def __call__(
        self,
        response: str | None,
        test_list: List[str] = [],
        setup_code: str = "",
        skip_parse: bool = True,
    ) -> CodeEvaluationResult:
        result = CodeEvaluationResult(
            parsed_response=None, is_formatted=True, can_compile=True,
            pass_rate=0.0, tests_passed=0, tests_evaluated=0,
            tests_total=len(test_list), test_errors=[],
        )

        program = response if skip_parse else self.parse_response(response)
        if program is None:
            result["is_formatted"] = False
            result["can_compile"] = False
            return result

        result["parsed_response"] = program
        test_runner_code = _create_test_runner_code(setup_code, program, test_list, self.max_failures)
        code_run_result = _run_code_subprocess(test_runner_code, timeout=self.timeout,
                                               memory_limit=self.memory_per_worker, debug=self.debug)

        result["can_compile"] = code_run_result.compiled
        if not code_run_result.compiled:
            result["test_errors"].append("MasterError: CompilationError")
        if code_run_result.timeout:
            result["test_errors"].append("MasterError: TimeoutError")
        if code_run_result.oom:
            result["test_errors"].append("MasterError: OOMError")
        if not code_run_result.success:
            result["test_errors"].append("MasterError: UnknownError: " + str(code_run_result.stdout.get("raw", "")))

        result["tests_evaluated"] = code_run_result.stdout.get("tests_evaluated", 0)
        result["tests_passed"] = code_run_result.stdout.get("tests_passed", 0)
        result["pass_rate"] = (result["tests_passed"] / result["tests_total"]) if result["tests_total"] > 0 else 0.0
        result["test_errors"] += code_run_result.stdout.get("test_errors", [])

        return result

    def batch_evaluate(self, calls: list[dict[str, Any]]) -> list[CodeEvaluationResult]:
        if not calls:
            return []
        results: List[Any] = [None] * len(calls)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_index = {executor.submit(self.__call__, **call): idx for idx, call in enumerate(calls)}
            pbar = tqdm(total=len(future_to_index), desc="Evaluating responses")
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
                pbar.update(1)
            pbar.close()
        return results

    def parse_response(self, response: str) -> str | None:
        blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE)
        if not blocks:
            return None
        cleaned = [b.strip() for b in blocks if b.strip()]
        return "\n\n".join(cleaned) if cleaned else None

    def check_compile(self, response: str) -> bool:
        program = self.parse_response(response)
        if program is None:
            return False
        return _run_code_subprocess(program, timeout=self.timeout, memory_limit=self.memory_per_worker).compiled

    def extract_function(self, code_str: str, func_name: str) -> str:
        try:
            tree = ast.parse(code_str)
        except Exception:
            return ""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return ast.unparse(node)
        return ""

    def extract_function_parent(self, code_str: str, func_name: str) -> str | None:
        try:
            tree = ast.parse(code_str)
        except Exception:
            return None

        class FunctionParentExtractor(ast.NodeVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.class_name = None
                self.current_class = None

            def visit_ClassDef(self, node):
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node):
                if node.name == self.target_name:
                    self.class_name = self.current_class
                self.generic_visit(node)

        extractor = FunctionParentExtractor(func_name)
        extractor.visit(tree)
        return extractor.class_name
