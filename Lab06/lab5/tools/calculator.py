"""Calculator tool — safe AST-based math expression evaluator."""
import ast
import math
import operator


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
}
_ALLOWED_FUNCS = {
    name: getattr(math, name) for name in [
        "sqrt", "log", "log2", "log10", "exp",
        "sin", "cos", "tan", "asin", "acos", "atan",
        "floor", "ceil", "fabs", "factorial",
        "degrees", "radians",
    ]
}
_ALLOWED_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})

_MAX_EXPRESSION_LEN = 300
_MAX_AST_NODES = 80
_MAX_ABS_NUMBER = 10 ** 12
_MAX_POWER_EXPONENT = 12
_MAX_FACTORIAL_ARG = 170


def _check_number(value):
    if isinstance(value, (int, float)) and abs(value) <= _MAX_ABS_NUMBER:
        return value
    raise ValueError("Number is too large.")


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return _check_number(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _eval(node.left)
        right = _eval(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POWER_EXPONENT:
            raise ValueError("Exponent is too large.")
        return _check_number(_ALLOWED_BINOPS[op_type](left, right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY:
            raise ValueError(f"Unsupported unary op: {op_type.__name__}")
        return _ALLOWED_UNARY[op_type](_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Unknown name: {node.id}")
    if isinstance(node, ast.Attribute):
        # Allow math.<name>
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and node.attr in _ALLOWED_FUNCS
        ):
            return _ALLOWED_FUNCS[node.attr]
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "math"
            and node.attr in _ALLOWED_NAMES
        ):
            return _ALLOWED_NAMES[node.attr]
        raise ValueError(f"Disallowed attribute: {ast.dump(node)}")
    if isinstance(node, ast.Call):
        if node.keywords:
            raise ValueError("Keyword arguments are not supported.")
        func = _eval(node.func) if isinstance(node.func, ast.Attribute) \
            else _ALLOWED_FUNCS.get(
                node.func.id if isinstance(node.func, ast.Name) else None
            )
        if func is None:
            raise ValueError("Disallowed function call.")
        args = [_eval(a) for a in node.args]
        if func is math.factorial and (
            len(args) != 1 or not isinstance(args[0], int)
            or args[0] < 0 or args[0] > _MAX_FACTORIAL_ARG
        ):
            raise ValueError("factorial expects an integer from 0 to 170.")
        return _check_number(func(*args))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculator(expression):
    """Evaluate a math expression safely.

    Supports: + - * / // % ** , parentheses, math.<func> and bare
    function names (sqrt, log, sin, cos, ...), constants pi/e/tau,
    abs/round/min/max.
    """
    if not isinstance(expression, str) or not expression.strip():
        return {"error": "Empty expression."}
    if len(expression) > _MAX_EXPRESSION_LEN:
        return {"expression": expression, "error": "Expression is too long."}
    try:
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
            return {"expression": expression, "error": "Expression is too complex."}
        result = _eval(tree)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "Evaluate a math expression. Supports +, -, *, /, //, %, **, "
            "parentheses, math functions (sqrt, log, sin, cos, ...), "
            "and constants (pi, e). Use this for any arithmetic the user "
            "asks about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "Math expression, e.g. '2 + 2 * sqrt(16)' or "
                        "'(15/100) * 240'."
                    ),
                },
            },
            "required": ["expression"],
        },
    },
}
