"""Sample module with a known, deterministic call structure.

Used by tests/test_call_graph.py to verify that CallGraphAnalyzer
correctly detects caller/callee relationships.

Call graph (expected):
    main  →  greet, compute
    greet →  format_name
    compute → add, multiply
    add   →  (nothing)
    multiply → (nothing)
    format_name → (nothing)
    orphan → (nothing, never called)
"""


def format_name(name: str) -> str:
    return name.strip().title()


def greet(name: str) -> str:
    return f"Hello, {format_name(name)}!"


def add(a: int, b: int) -> int:
    return a + b


def multiply(a: int, b: int) -> int:
    return a * b


def compute(x: int, y: int) -> int:
    total = add(x, y)
    product = multiply(x, y)
    return total + product


def main() -> None:
    print(greet("world"))
    result = compute(3, 4)
    print(result)


def orphan() -> str:
    """This function is defined but never called by anything in this module."""
    return "I am never called"
