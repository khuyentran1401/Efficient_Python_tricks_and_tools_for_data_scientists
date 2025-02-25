---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

## Typing

+++

typing is a Python module that allows developers to specify the types of inputs to make sure the input types are correct.

+++

### typing.Callable: Specify an Input is of Type Function

```{code-cell} ipython3
:tags: [hide-cell]

!pip install mypy
```

If you want to specify an input is of type function, use `typing.Callable`.  

```{code-cell} ipython3
%%writefile callable_example.py 
from typing import Callable

def multiply(x: float, y: float) -> float:
    return x * y

def multiply_then_divide_by_two(multiply_func: Callable[[float, float], float], x: float, y: float) -> float:
    return multiply_func(x, y) / 2

res = multiply_then_divide_by_two(multiply, 2, 3)
```

```bash
$ mypy callable_example.py
```

+++

`Callable` can now be used static type checker such as mypy to check if the input is indeed a function. 

```{code-cell} ipython3
:tags: [hide-input]

!mypy callable_example.py
```

### Use Python Class as a Type Hint

+++

In the code below, `Orange` and `Apple` are subclasses of `Fruit`. How do we use type hint to specify that `fruit_type` in `make_fruit` should be a subclass of `Fruit`? 

Using a parent class as a type hint will give you a type error when using mypy. 

```{code-cell} ipython3
%%writefile type_example_wrong.py
class Fruit:
    def __init__(self, taste: str) -> None:
        self.taste = taste 

class Orange(Fruit):
    ... 

class Apple(Fruit):
    ... 

def make_fruit(fruit_type: Fruit, taste: str):
    return fruit_type(taste=taste)

orange = make_fruit(Orange, "sour")
```

```bash
$ mypy type_example_wrong.py
```

```{code-cell} ipython3
:tags: [remove-input]

!mypy type_example_wrong.py
```

Use `typing.Type` instead.

```{code-cell} ipython3
%%writefile type_example_right.py
from typing import Type 

class Fruit:
    def __init__(self, taste: str) -> None:
        self.taste = taste 

class Orange(Fruit):
    ... 

class Apple(Fruit):
    ... 

def make_fruit(fruit_type: Type[Fruit], taste: str):
    return fruit_type(taste=taste)

orange = make_fruit(Orange, "sour")
```

```bash
$ mypy type_example_right.py
```

```{code-cell} ipython3
:tags: [remove-input]

!mypy type_example_right.py
```

### typing.Annotated: Add Metadata to Your Typehint

+++

If you want to add metadata to your typehint such as units of measurement, use `typing.Annotated`.

```{code-cell} ipython3
%%writefile typing_annotated.py 
from typing import Annotated


def get_height_in_feet(height: Annotated[float, "meters"]):
    return height * 3.28084


print(get_height_in_feet(height=1.6))
```

`Annotated[T, x]` allows static typechecking such as mypy to check `T` while safely ignoring `x`. 

```{code-cell} ipython3
!mypy typing_annotated.py 
```

This method is available for Python 3.9 and above. 

+++

### typing.final: Declare That a Method Should Not Be Overridden

+++

If you want to declare that some methods shouldn't be overridden by subclasses, use the decorator `typing.final`.  

```{code-cell} ipython3
%%writefile typing_final.py
from typing import final 

class Dog:
    @final 
    def bark(self) -> None:
        print("Woof")

class Dachshund(Dog):
    def bark(self) -> None:
        print("Ruff")

bim = Dachshund()
bim.bark()
```

```bash
$ mypy typing_final.py
```

```{code-cell} ipython3
:tags: [remove-input]

!mypy typing_final.py
```

This method is available for Python 3.8 and above. 

+++

### typing.Literal: Specify the Possible Values for a Function Parameter 

+++

If you want to use type hints to check that a variable or a function parameter is in a set of literal values, use `typing.Literal`. 

In the example below, since `grape` is not in the set of literal values, mypy raises an error. 

```{code-cell} ipython3
%%writefile typing_literal.py
from typing import Literal


def get_price(fruit: Literal["apple", "orange"]):
    if fruit == "apple":
        return 1
    else:  # if it is orange
        return 2


get_price("grape")
```

```bash
$ mypy typing_literal.py
```

```{code-cell} ipython3
:tags: [remove-input]

!mypy typing_literal.py 
```

This method is available in Python 3.8 and above. 

+++

### typing.TypeVar: Flexible Typing for Context-Dependent Types

+++

If you have multiple functions with a shared purpose but differing only in element types, group them into one function to improve code readability and scalability. 

```{code-cell} ipython3
def last_int(l: list[int]) -> int:
    return l[-1]

def last_str(l: list[str]) -> str:
    return l[-1]
```

Type variables allow you to create generic code that can adapt to various types based on the context in which it is invoked.

In the first call, the list `list(range(10))` contains integers, so the type of `T` is inferred to be `int`. In the second example call, the list `list("abc")` contains strings, so the type of `T` is inferred to be `str`.

```{code-cell} ipython3
%%writefile typevar_example.py 
from typing import TypeVar


T = TypeVar("T")


def last(l: list[T]) -> T:
    return l[-1]


if __name__ == "__main__":
    last(list(range(10)))
    last(list("abc"))
```

```bash
$ mypy typevar_example.py
```

```{code-cell} ipython3
:tags: [remove-input]

!mypy typevar_example.py
```

### Write Union Types with X|Y

+++

`typing.Union[X, Y]` is used to declare that a variable can be either `X` or `Y`. In Python 3.10 and above, you can replace `Union[X, Y]` with `X|Y`. 

```{code-cell} ipython3
# Before Python 3.10
from typing import Dict, Union


def get_price(grocery: Dict[str, Union[int, float]]):
    return grocery.values()


grocery = {"apple": 3, "orange": 2.5}
get_price(grocery)
 
```

```{code-cell} ipython3
# In Python 3.10
def get_price(grocery: dict[str, int | float]):
    return grocery.values()


grocery = {"apple": 3, "orange": 2.5}
get_price(grocery)
```
