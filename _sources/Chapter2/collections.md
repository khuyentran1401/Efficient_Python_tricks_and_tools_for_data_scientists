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

```{code-cell} ipython3
:tags: [remove-cell]

import warnings
warnings.filterwarnings("ignore")
```

## Collections

+++

[collections](https://docs.python.org/3/library/collections.html) is a built-in Python library to deal with Python dictionary efficiently. This section will show you some useful methods of this module. 

+++

![](../img/collections.png)

+++

### collections.Counter: Count The Occurrences of Items in a List

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import HTML

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/03vtzKflwzI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

Counting the occurrences of each item in a list using a for-loop is slow and inefficient. 

```{code-cell} ipython3
char_list = ["a", "b", "c", "a", "d", "b", "b"]
```

```{code-cell} ipython3
def custom_counter(list_: list):
    char_counter = {}
    for char in list_:
        if char not in char_counter:
            char_counter[char] = 1
        else:
            char_counter[char] += 1

    return char_counter


custom_counter(char_list)
```

Using `collections.Counter` is more efficient, and all it takes is one line of code!

```{code-cell} ipython3
from collections import Counter

Counter(char_list)
```

In my experiment, using `Counter` is more than 2 times faster than using a custom counter.

```{code-cell} ipython3
from timeit import timeit
import random

random.seed(0)
num_list = [random.randint(0, 22) for _ in range(1000)]

numExp = 100
custom_time = timeit("custom_counter(num_list)", globals=globals())
counter_time = timeit("Counter(num_list)", globals=globals())
print(custom_time / counter_time)
```

### namedtuple: A Lightweight Python Structure to Mange your Data

+++

If you need a small class to manage data in your project, consider using namedtuple.

`namedtuple` object is like a tuple but can be used as a normal Python class.

In the code below, I use `namedtuple` to create a `Person` object with attributes `name` and `gender`.

```{code-cell} ipython3
from collections import namedtuple

Person = namedtuple("Person", "name gender")

oliver = Person("Oliver", "male")
khuyen = Person("Khuyen", "female")
```

```{code-cell} ipython3
oliver
```

```{code-cell} ipython3
khuyen
```

Just like Python class,  you can access attributes of `namedtuple` using `obj.attr`.

```{code-cell} ipython3
oliver.name
```

### Defaultdict: Return a Default Value When a Key is Not Available

```{code-cell} ipython3
:tags: [remove-input]

from IPython.display import HTML

# Youtube
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/4ivlsfA9xos" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
```

If you want to create a Python dictionary with default value, use `defaultdict`. When calling a key that is not in the dictionary, the default value is returned.

```{code-cell} ipython3
from collections import defaultdict

classes = defaultdict(lambda: "Outside")
classes["Math"] = "B23"
classes["Physics"] = "D24"
```

```{code-cell} ipython3
classes["Spanish"]
```

### Defaultdict: Create a Dictionary with Values that are List

+++

If you want to create a dictionary with the values that are list, the cleanest way is to pass a list class to a `defaultdict`.

```{code-cell} ipython3
from collections import defaultdict

# Instead of this
food_price = {"apple": [], "orange": []}

# Use this
food_price = defaultdict(list)

for i in range(1, 4):
    food_price["apple"].append(i)
    food_price["orange"].append(i)

print(food_price.items())
```

### OrderedDict: Create an Ordered Python Dictionary 

+++

Comparing two Python dictionaries ignores the order of items. 

```{code-cell} ipython3
unordered1 = {'a': 1, 'b': 2, 'c': 3}
unordered2 = {'b': 2, 'a': 1, 'c': 3}
unordered1 == unordered2
```

If you want to consider the order of items, use `OrderedDict` instead.

```{code-cell} ipython3
from collections import OrderedDict

ordered1 = OrderedDict({'a': 1, 'b': 2, 'c': 3})
ordered2 = OrderedDict({'b': 2, 'a': 1, 'c': 3})
ordered1 == ordered2
```

### ChainMap: Combine Multiple Dictionaries into One Unit

+++

If you want to combine multiple dictionaries into one unit, `collections.ChainMap` is a good option. `ChainMap` allows you to organize and get the keys or values across different dictionaries.

```{code-cell} ipython3
from collections import ChainMap
```

```{code-cell} ipython3
fruits = {'apple': 2, 'tomato': 1}
veggies = {'carrot': 3, 'tomato': 1}
food = ChainMap(fruits, veggies) 
```

```{code-cell} ipython3
food.maps # get all contents
```

```{code-cell} ipython3
list(food.keys()) # Get keys
```

```{code-cell} ipython3
list(food.values()) # Get values
```
