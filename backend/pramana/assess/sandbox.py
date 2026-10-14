"""Run generator-written checking code for numerical questions.

The generator supplies Python defining `exact()` (the answer computed symbolically or with
exact arithmetic) and optionally `simulate(rng, n)` (a Monte Carlo estimate). The code is
model-written, so it is screened by an AST allow-list and run in a separate interpreter
with a timeout. This guards against accidents, not a determined attacker.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {"math", "fractions", "itertools", "random", "statistics", "sympy", "functools", "collections", "decimal"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "getattr",
                   "setattr", "delattr", "vars", "breakpoint", "exit", "quit"}

_HARNESS = '''
import json, random, math
_out = {}
try:
    _v = exact()
    _out["exact"] = float(_v)
except Exception as _e:
    _out["exact_error"] = repr(_e)[:300]
if "simulate" in globals():
    try:
        _n = 200000
        _out["simulated"] = float(simulate(random.Random(12345), _n))
        _out["n"] = _n
    except Exception as _e:
        _out["simulate_error"] = repr(_e)[:300]
print("PRAMANA_RESULT " + json.dumps(_out))
'''


def screen(code: str) -> str | None:
    """Return a reason to refuse the code, or None if it passes the allow-list."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad = [a.name for a in node.names if a.name.split(".")[0] not in ALLOWED_IMPORTS]
            if bad:
                return f"import not allowed: {bad}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import not allowed: {node.module}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"name not allowed: {node.id}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"attribute not allowed: {node.attr}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == "exact" for n in tree.body):
        return "code must define exact()"
    return None


def run_check(code: str, timeout: float = 30.0) -> dict:
    """Execute checking code. Returns {"exact": x, "simulated": y, "n": N} or {"error": reason}."""
    reason = screen(code)
    if reason:
        return {"error": reason}
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "check.py"
        script.write_text(code + "\n" + _HARNESS, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True,
                                  timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return {"error": f"timed out after {timeout:.0f}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("PRAMANA_RESULT "):
            return json.loads(line[len("PRAMANA_RESULT "):])
    return {"error": (proc.stderr or "no result").strip()[-300:]}
