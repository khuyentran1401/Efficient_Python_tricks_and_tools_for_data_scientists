
# callable_example.py 

from typing import Callable

def multiply(x: float, y: float) -> float:
    return x * y

def multiply_then_divide_by_two(multiply_func: Callable[[float, float], float], x: float, y: float) -> float:
    return multiply_func(x, y) / 2

res = multiply_then_divide_by_two(multiply, 2, 3)
