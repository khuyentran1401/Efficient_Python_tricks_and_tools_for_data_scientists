---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Operator

+++

[operator](https://docs.python.org/3/library/operator.html) is a built-in Python library that exports a set of efficient functions corresponding to the intrinsic operators of Python. This section will show you some useful methods of this module. 

+++

### operator.itemgetter: Get Multiple Items From a List or Dictionary

+++

Normally, to access multiple indices from a list, you need to use list comprehension: 

```{code-cell} ipython3
fruits = ["apple", "orange", "banana", "grape"]

[fruit for fruit in fruits if fruits.index(fruit) in [1, 3]]
```

To do the same thing with simpler syntax, use `operator.itemgetter` instead: 

```{code-cell} ipython3
from operator import itemgetter

itemgetter(1, 3)(fruits)
```
