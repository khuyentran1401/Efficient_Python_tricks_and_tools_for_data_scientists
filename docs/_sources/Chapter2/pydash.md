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

## Pydash

+++

[Pydash](https://pydash.readthedocs.io/en/latest/) is the kitchen sink of Python utility libraries for doing “stuff” in a functional way.

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pydash 
```

### Work with List

+++

#### pydash.flatten: Flatten a Nested Python List

+++

If you want to flatten a nested Python list, use the `flatten` method:

```{code-cell} ipython3
from pydash import py_

a = [[1, 2], [3, 4, 5]]
py_.flatten(a)
```

####  pydash.flatten_deep: Flatten a Deeply Nested Python List

+++

If your list is deeply nested, use `flatten_deep` instead:

```{code-cell} ipython3
b = [[1, 2, [4, 5]], [6, 7]]
py_.flatten_deep(b)
```

#### pydash.chunk: Split Elements into Groups

+++

If you can flatten a list, can you also turn a flattened list into a nested one? Yes, that could be done with the `chunk` method:

```{code-cell} ipython3
a = [1, 2, 3, 4, 5]
py_.chunk(a, 2)
```

### Work with Dictionary

+++

#### Omit Dictionary’s Attribute

+++

To omit an attribute from the dictionary, we can use the `omit` method:

```{code-cell} ipython3
fruits = {"name": "apple", "color": "red", "taste": "sweet"}
py_.omit(fruits, "name")
```

#### Get Nested Dictionary’s Attribute

+++

How do you get the price of an apple from Walmart that is in season in a nested dictionary like below?

```{code-cell} ipython3
apple = {
    "price": {
        "in_season": {"store": {"Walmart": [2, 4], "Aldi": 1}},
        "out_of_season": {"store": {"Walmart": [3, 5], "Aldi": 2}},
    }
}
```

Normally, you need to use a lot of brackets to get that information:

```{code-cell} ipython3
apple["price"]["in_season"]["store"]["Walmart"]
```

Wouldn’t it be nice if you could use the dot notation instead of brackets? That could be done with the `get` method:

```{code-cell} ipython3
py_.get(apple, "price.in_season.store.Walmart")
```

Cool! You can also get the element in an array using `[index]` :

```{code-cell} ipython3
py_.get(apple, "price.in_season.store.Walmart[0]")
```

### Work with List of Dictionaries

+++

#### Find Item Index Using a Function

+++

`list.index(element)` allows you to get the index of the specified element in a list. However, you cannot get the index using a function.   

```{code-cell} ipython3
fruits = [
    {"name": "apple", "price": 2},
    {"name": "orange", "price": 2},
    {"name": "grapes", "price": 4},
]
```

```{code-cell} ipython3
fruits.index({"name": "apple", "price": 2})
```

```{code-cell} ipython3
filter_fruits = lambda fruit: fruit["name"] == "apple"

fruits.index(filter_fruits)
```

To get the index of an element in a list using a function, use the `find_index` method instead:

```{code-cell} ipython3
py_.find_index(fruits, filter_fruits)
```

#### Find Objects With Matching Style

+++

The `find_index` method allows you to get the index of the object that matches a certain pattern. But what if you want to get the items in a list instead of the index?
That could be done with the `filter` method:

```{code-cell} ipython3
from pydash import py_
```

```{code-cell} ipython3
fruits = [
    {"name": "apple", "price": 2},
    {"name": "orange", "price": 2},
    {"name": "grapes", "price": 4},
]
```

```{code-cell} ipython3
py_.filter_(fruits, {"name": "apple"})
```

#### Get Nested Object Value

+++

Sometimes your list of dictionaries can be nested like below. How can you get the second price of each `apple`?

```{code-cell} ipython3
l = [
    {"apple": {"price": [0, 1], "color": "red"}},
    {"apple": {"price": [2, 3], "color": "green"}},
]
```

That is when the `map_` method comes in handy.

```{code-cell} ipython3
py_.map_(l, "apple.price[1]")
```

### Work with Functions

+++

#### Execute a Function n Times

+++

You can execute a function n times using the `times` method. This method is a good alternative to a for loop.

```{code-cell} ipython3
py_.times(4, lambda: "I have just bought some apple")
```

```{code-cell} ipython3
py_.times(4, lambda i: f"I have just bought {i} apple")
```

### Chaining

+++

#### Pydash’s Methods

+++

Sometimes you might want to apply several methods to an object. Instead of writing several lines of code, can you apply all methods at once?

That is when method chaining comes in handy. To apply method chaining in an object, use the `chain` method:

```{code-cell} ipython3
fruits = ["apple", "orange", "grapes"]
```

```{code-cell} ipython3
(py_.chain(fruits).without("grapes").reject(lambda fruit: fruit.startswith("a")))
```

Note that running the code above will not give us the value.

+++

Only when we add .value() to the end of the chain, the final value is computed:

```{code-cell} ipython3
(
    py_.chain(fruits)
    .without("grapes")
    .reject(lambda fruit: fruit.startswith("a"))
    .value()
)
```

This is called [lazy evaluation](https://www.tutorialspoint.com/functional_programming/functional_programming_lazy_evaluation.htm). Lazy evaluation holds the evaluation of an expression until its value is needed, which avoids repeated evaluation.

+++

#### Customized Methods

+++

If you want to use your own methods instead of pydash’s methods, use the `map` method:

```{code-cell} ipython3
def get_price(fruit):
    prices = {"apple": 2, "orange": 2, "grapes": 4}
    return prices[fruit]
```

```{code-cell} ipython3
total_price = py_.chain(fruits).map(get_price).sum()
total_price.value()
```

#### Planting a Value

+++

To replace the initial value of a chain with another value, use the `plant` method:

```{code-cell} ipython3
total_price.plant(["apple", "orange"]).value()
```

Cool! We replace `['apple', 'orange', 'grapes']` with `['apple', 'orange']` while using the same chain!
