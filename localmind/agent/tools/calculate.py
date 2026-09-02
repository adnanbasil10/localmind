"""`calculate`: a genuinely sandboxed arithmetic evaluator. Never `eval()`.

The evaluator parses the expression to an AST (`ast.parse` only *parses*; it
never executes) and then walks it with an **allowlist** of node types, operators,
names and functions. Anything not on the allowlist is rejected before any value
is computed. There is no `eval`, no `exec`, no `compile` of user text, and no
`__builtins__` reachable from the walker at all -- there is no name resolution
path to Python objects, because names are looked up in a fixed dict of floats.

Denial of service is a real attack here too, so the walker enforces, in order:
character cap, node cap, depth cap, a pre-check on every `**` (rejecting
`9**9**9` *before* computing it), a pre-check on every allowlisted function that
hides an exponent (`factorial`, `pow`, and `round`'s `ndigits` -- see
`_guarded_round`), and a magnitude cap re-checked after every binary operation.

The pre-checks are load-bearing rather than belt-and-braces: CPython's bigint
arithmetic runs in C with the GIL held, so the tool timeout in `base.py` cannot
preempt an expensive call once it has started. Every DoS guard here therefore
refuses *before* evaluation; none of them rely on being able to interrupt.

`safe_eval` is importable on its own; `CalculateTool` is the agent-facing wrapper.
"""

from __future__ import annotations

import ast
import math
import unicodedata
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from localmind.agent.tools.base import Tool, ToolExecutionError

__all__ = ["CalculateTool", "UnsafeExpressionError", "safe_eval"]

MAX_EXPRESSION_CHARS = 256
MAX_NODES = 128
MAX_DEPTH = 16
MAX_POW_EXPONENT = 64
MAX_INT_BITS = 1024
MAX_FACTORIAL = 20
MAX_CALL_ARGS = 4
MAX_ROUND_NDIGITS = MAX_POW_EXPONENT
"""`round(x, ndigits)` hides a `**`, so `ndigits` gets the `**` bound.

CPython's `int.__round__` evaluates `10 ** abs(ndigits)` internally. That makes
`ndigits` an exponent wearing a different hat, and it is bounded by the same
number as an explicit `**` exponent rather than by a limit of its own.
"""


class UnsafeExpressionError(ValueError):
    """The expression is rejected. The message names the specific rule that fired."""


Number = int | float

_BIN_OPS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
}

_UNARY_OPS: dict[type[ast.unaryop], str] = {ast.UAdd: "+", ast.USub: "-"}

_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


def _guarded_factorial(n: Number) -> int:
    if not float(n).is_integer() or n < 0:
        raise UnsafeExpressionError("factorial requires a non-negative integer")
    if n > MAX_FACTORIAL:
        raise UnsafeExpressionError(f"factorial argument exceeds {MAX_FACTORIAL}")
    return math.factorial(int(n))


def _guarded_pow(base: Number, exponent: Number) -> Number:
    _check_pow(base, exponent)
    return _check_magnitude(base**exponent)


def _guarded_round(value: Number, ndigits: Number | None = None) -> Number:
    """Bound `ndigits` *before* the call, because nothing can stop it afterwards.

    `int.__round__` computes `10 ** (-ndigits)` internally, so `round(2, -10**7)`
    -- 16 characters, 6 AST nodes, inside every character/node/depth cap, with no
    `**` on the expensive operand -- spends ~11 s building a multi-megabyte
    integer. `_check_magnitude` cannot help: it runs on the *result*, which is a
    tidy `0`, long after the giant intermediate has been built and discarded.

    Nor can the tool timeout help. That work happens entirely in C with the GIL
    held, so `base.call_with_timeout`'s watchdog thread is never scheduled and
    `CalculateTool(timeout_s=1.0)` returns `ok=True` after 11 seconds. Refusal
    before evaluation is the only defence that actually works here; see
    `tests/test_agent.py::test_calculate_refuses_gil_holding_work_before_running_it`.
    """
    if ndigits is None:
        return _check_magnitude(round(value))
    # `abs()` on a bounded int is O(1) and never converts to float, so this
    # comparison cannot itself be the expensive step.
    if abs(ndigits) > MAX_ROUND_NDIGITS:
        raise UnsafeExpressionError(
            f"round() ndigits magnitude exceeds the sandbox limit of {MAX_ROUND_NDIGITS} "
            f"(denial-of-service guard: ndigits is a power-of-ten exponent)"
        )
    if not isinstance(ndigits, int):
        raise UnsafeExpressionError("round() ndigits must be an integer")
    return _check_magnitude(round(value, ndigits))


_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": _guarded_round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "fabs": math.fabs,
    "hypot": math.hypot,
    "gcd": math.gcd,
    "factorial": _guarded_factorial,
    "pow": _guarded_pow,
}


def _check_magnitude(value: Any) -> Number:
    """Reject results that are too large to be worth computing or printing."""
    if isinstance(value, bool):  # bool is an int subclass; not a number we want
        raise UnsafeExpressionError("boolean values are not supported")
    if isinstance(value, int):
        if value.bit_length() > MAX_INT_BITS:
            raise UnsafeExpressionError(
                f"integer result exceeds {MAX_INT_BITS} bits (denial-of-service guard)"
            )
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise UnsafeExpressionError("result is NaN")
        if math.isinf(value):
            raise UnsafeExpressionError("result overflowed to infinity")
        return value
    raise UnsafeExpressionError(f"unsupported result type {type(value).__name__}")


def _check_pow(base: Number, exponent: Number) -> None:
    """Reject huge-exponent DoS (`9**9**9`) *before* the power is computed."""
    if isinstance(exponent, float) and not math.isfinite(exponent):
        raise UnsafeExpressionError("non-finite exponent")
    if abs(base) <= 1:
        if abs(exponent) > 1e6:
            raise UnsafeExpressionError("exponent magnitude exceeds the sandbox limit")
        return
    if exponent > MAX_POW_EXPONENT:
        raise UnsafeExpressionError(
            f"exponent {exponent} exceeds the sandbox limit of {MAX_POW_EXPONENT}"
        )
    if exponent < -MAX_POW_EXPONENT * 16:
        raise UnsafeExpressionError("exponent magnitude exceeds the sandbox limit")
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and base.bit_length() * exponent > MAX_INT_BITS
    ):
        raise UnsafeExpressionError(
            f"result would exceed {MAX_INT_BITS} bits (denial-of-service guard)"
        )


class _SafeEvaluator(ast.NodeVisitor):
    """Allowlist walker. Any node type without an explicit visitor is rejected."""

    def __init__(self) -> None:
        self.nodes = 0

    # -- rejection default -------------------------------------------------------------

    def generic_visit(self, node: ast.AST) -> Number:  # type: ignore[override]
        raise UnsafeExpressionError(
            f"{type(node).__name__} is not permitted in the calculator sandbox"
        )

    def _visit(self, node: ast.AST, depth: int) -> Number:
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise UnsafeExpressionError(f"expression exceeds {MAX_NODES} AST nodes")
        if depth > MAX_DEPTH:
            raise UnsafeExpressionError(f"expression nesting exceeds depth {MAX_DEPTH}")

        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.BinOp):
            return self._binop(node, depth)
        if isinstance(node, ast.UnaryOp):
            return self._unaryop(node, depth)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Call):
            return self._call(node, depth)
        return self.generic_visit(node)

    # -- allowed node types ------------------------------------------------------------

    @staticmethod
    def _constant(node: ast.Constant) -> Number:
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise UnsafeExpressionError(
                f"only int and float literals are permitted, got {type(value).__name__}"
            )
        return _check_magnitude(value)

    def _binop(self, node: ast.BinOp, depth: int) -> Number:
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise UnsafeExpressionError(f"operator {op_type.__name__} is not permitted")
        left = self._visit(node.left, depth + 1)
        right = self._visit(node.right, depth + 1)
        try:
            if op_type is ast.Pow:
                _check_pow(left, right)
                return _check_magnitude(left**right)
            if op_type is ast.Add:
                return _check_magnitude(left + right)
            if op_type is ast.Sub:
                return _check_magnitude(left - right)
            if op_type is ast.Mult:
                return _check_magnitude(left * right)
            if op_type is ast.Div:
                return _check_magnitude(left / right)
            if op_type is ast.FloorDiv:
                return _check_magnitude(left // right)
            return _check_magnitude(left % right)
        except ZeroDivisionError as exc:
            raise UnsafeExpressionError("division by zero") from exc
        except OverflowError as exc:
            raise UnsafeExpressionError("numeric overflow") from exc

    def _unaryop(self, node: ast.UnaryOp, depth: int) -> Number:
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise UnsafeExpressionError(f"unary operator {op_type.__name__} is not permitted")
        operand = self._visit(node.operand, depth + 1)
        return _check_magnitude(-operand if op_type is ast.USub else +operand)

    @staticmethod
    def _name(node: ast.Name) -> Number:
        if node.id not in _CONSTANTS:
            raise UnsafeExpressionError(
                f"name {node.id!r} is not defined in the sandbox "
                f"(allowed: {', '.join(sorted(_CONSTANTS))})"
            )
        return _CONSTANTS[node.id]

    def _call(self, node: ast.Call, depth: int) -> Number:
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("only direct calls to allowlisted functions are permitted")
        if node.keywords:
            raise UnsafeExpressionError("keyword arguments are not permitted")
        if node.func.id not in _FUNCTIONS:
            raise UnsafeExpressionError(f"function {node.func.id!r} is not on the allowlist")
        if len(node.args) > MAX_CALL_ARGS:
            raise UnsafeExpressionError(f"at most {MAX_CALL_ARGS} arguments are permitted")
        args: list[Number] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise UnsafeExpressionError("starred arguments are not permitted")
            args.append(self._visit(arg, depth + 1))
        fn = _FUNCTIONS[node.func.id]
        try:
            return _check_magnitude(fn(*args))
        except UnsafeExpressionError:
            raise
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise UnsafeExpressionError(f"{node.func.id}: {exc}") from exc


def safe_eval(expression: str) -> Number:
    """Evaluate an arithmetic expression under the sandbox. Raises `UnsafeExpressionError`.

    Security properties: no `eval`/`exec`/`compile` of the input; attribute
    access, subscripting, comprehensions, lambdas, f-strings, imports and every
    other node type are rejected by allowlist default; names resolve only to a
    fixed dict of floats, so there is no path to a Python object and therefore no
    dunder traversal; `**` is bounded before evaluation.
    """
    if not isinstance(expression, str):
        raise UnsafeExpressionError("expression must be a string")
    text = expression.strip()
    if not text:
        raise UnsafeExpressionError("empty expression")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise UnsafeExpressionError(f"expression exceeds {MAX_EXPRESSION_CHARS} characters")
    if "\x00" in text:
        raise UnsafeExpressionError("null byte in expression")
    if "__" in unicodedata.normalize("NFKC", text):
        # Belt and braces: dunder traversal is already impossible (no Attribute node,
        # no name resolution to objects), but refusing the substring outright makes
        # the property obvious to a reader and to an auditor.
        #
        # Normalised first because Python normalises identifiers with NFKC, so a
        # fullwidth `U+FF3F` spelling of `__import__` reaches the parser as the
        # dunder while defeating a raw substring test. CPython 3.11 happens to
        # reject U+FF3F at tokenize time, so this was never a live bypass -- but a
        # check that only works because of an accident elsewhere is not a check.
        raise UnsafeExpressionError("dunder names are not permitted")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"syntax error: {exc.msg}") from exc
    except (ValueError, MemoryError, RecursionError) as exc:  # pragma: no cover - defensive
        raise UnsafeExpressionError(f"could not parse expression: {type(exc).__name__}") from exc

    evaluator = _SafeEvaluator()
    try:
        return evaluator._visit(tree.body, 0)
    except RecursionError as exc:  # pragma: no cover - depth cap fires first
        raise UnsafeExpressionError("expression nesting is too deep") from exc


class CalculateArgs(BaseModel):
    expression: str = Field(
        description="Arithmetic expression, e.g. '(3 + 4) * sqrt(2)'. Sandboxed: "
        "numbers, + - * / // % **, and allowlisted math functions only.",
        max_length=MAX_EXPRESSION_CHARS,
    )
    precision: int = Field(default=6, ge=0, le=15, description="Decimal places for float results.")


class CalculateTool(Tool):
    """Deterministic, idempotent, sandboxed arithmetic."""

    name: ClassVar[str] = "calculate"
    description: ClassVar[str] = (
        "Evaluate an arithmetic expression in a sandbox (no eval, no imports, no "
        "attribute access). Supports + - * / // % ** and common math functions."
    )
    args_model: ClassVar[type[BaseModel]] = CalculateArgs
    idempotent: ClassVar[bool] = True

    def _call(self, args: CalculateArgs) -> dict[str, Any]:
        try:
            value = safe_eval(args.expression)
        except UnsafeExpressionError as exc:
            raise ToolExecutionError("invalid_input", f"rejected by sandbox: {exc}") from exc
        if isinstance(value, float):
            value = round(value, args.precision)
        return {"expression": args.expression, "result": value}
