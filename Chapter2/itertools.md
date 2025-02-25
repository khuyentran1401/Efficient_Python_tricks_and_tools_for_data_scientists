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

## Itertools

+++

![](../img/itertools.png)

+++

[itertools](https://docs.python.org/3/library/itertools.html) is a built-in Python library that creates iterators for efficient looping. This section will show you some useful methods of itertools.

+++

### itertools.combinations: A Better Way to Iterate Through a Pair of Values in a Python List

+++

If you want to iterate through a pair of values in a list and the order does not matter (`(a,b)` is the same as `(b, a)`), a naive approach is to use two for-loops.

```{code-cell} ipython3
num_list = [1, 2, 3]
```

```{code-cell} ipython3
(1, 2)
(1, 3)
(2, 3)
```

```{code-cell} ipython3
for i in num_list:
    for j in num_list:
        if i < j:
            print((i, j))
```

However, using two for-loops is lengthy and inefficient. Use `itertools.combinations` instead:

```{code-cell} ipython3
from itertools import combinations

comb = combinations(num_list, 2)  # use this
for pair in list(comb):
    print(pair)
```

### itertools.product: Nested For-Loops in a Generator Expression 

+++

Are you using nested for-loops to experiment with different combinations of parameters? 

```{code-cell} ipython3
params = {
    "learning_rate": [1e-1, 1e-2, 1e-3],
    "batch_size": [16, 32, 64],
}
for learning_rate in params["learning_rate"]:
    for batch_size in params["batch_size"]:
        combination = (learning_rate, batch_size)
        print(combination)
```

If so, use `itertools.product` instead.

`itertools.product` is more efficient than nested loop because `product(A, B)` returns the same as `((x,y) for x in A for y in B)`.

```{code-cell} ipython3
from itertools import product

params = {
    "learning_rate": [1e-1, 1e-2, 1e-3],
    "batch_size": [16, 32, 64],
}

for combination in product(*params.values()):
    print(combination)
```

### itertools.starmap: Apply a Function With More Than 2 Arguments to Elements in a List

+++

`map` is a useful method that allows you to apply a function to elements in a list. However, it can't apply a function with more than one argument to a list.

```{code-cell} ipython3
def multiply(x: float, y: float):
    return x * y
```

```{code-cell} ipython3
nums = [(1, 2), (4, 2), (2, 5)]
list(map(multiply, nums))
```

To apply a function with more than 2 arguments to elements in a list, use `itertools.starmap`. With `starmap`, elements in each tuple of the list `nums` are used as arguments for the function `multiply`.

```{code-cell} ipython3
from itertools import starmap

list(starmap(multiply, nums))
```

### itertools.compress: Filter a List Using Booleans

+++

Normally, you cannot filter a list using a list.

```{code-cell} ipython3
fruits = ["apple", "orange", "banana", "grape", "lemon"]
chosen = [1, 0, 0, 1, 1]
fruits[chosen]
```

To filter a list using a list of booleans, use `itertools.compress` instead 

```{code-cell} ipython3
from itertools import compress

list(compress(fruits, chosen))
```

### itertools.groupby: Group Elements in an Iterable by a Key

+++

If you want to group elements in a list by a key, use `itertools.groupby`. In the example below, I grouped elements in the list by the first element in each tuple. 

```{code-cell} ipython3
from itertools import groupby

prices = [("apple", 3), ("orange", 2), ("apple", 4), ("orange", 1), ("grape", 3)]

key_func = lambda x: x[0]

# Sort the elements in the list by the key
prices.sort(key=key_func)

# Group elements in the list by the key
for key, group in groupby(prices, key_func):
    print(key, ":", list(group))
```

### itertools.zip_longest: Zip Iterables of Different Lengths

+++

`zip` allows you to aggregate elements from each of the iterables. However, `zip` doesn't show all pairs of elements when iterables have different lengths. 

```{code-cell} ipython3
fruits = ["apple", "orange", "grape"]
prices = [1, 2]
```

```{code-cell} ipython3
list(zip(fruits, prices))
```

To aggregate iterables of different lengths, use `itertools.zip_longest`. This method will fill missing values with `fillvalue`.

```{code-cell} ipython3
from itertools import zip_longest
```

```{code-cell} ipython3
list(zip_longest(fruits, prices, fillvalue="-"))
```

### Python's dropwhile: A Clean Approach to Sequential Filtering

+++

Repeatedly checking elements in an iteration until a condition is met results in verbose and less readable code, especially when you want to skip elements at the beginning of a sequence.

```{code-cell} ipython3
text = "123ABC456"
alpha_part = []
found_alpha = False

for char in text:
    if found_alpha or not char.isdigit():
        found_alpha = True
        alpha_part.append(char)


print(''.join(alpha_part))
```

With `itertools.dropwhile`, you can elegantly skip elements from the beginning of an iteration until a condition becomes False, making your code more concise and functional.

```{code-cell} ipython3
from itertools import dropwhile

text = "123ABC456"
alpha_part = dropwhile(str.isdigit, text)

print(''.join(alpha_part))
```

In this example, `dropwhile` takes a predicate function (`str.isdigit`) and the text string. It skips digits until it finds a non-digit character, then returns the rest of the string. 

+++

### itertools.islice: Efficient Data Processing for Large Data Streams

+++

Working with large data streams or files can be challenging due to memory limitations. Index slicing is not feasible for extremely large data sets as it requires the entire list to reside in memory.

```python
# Load all log entries into memory as a list
large_log = [log_entry for log_entry in open("large_log_file.log")]

# Iterate over the first 100 entries of the list
for entry in large_log[:100]:
    process_log_entry(entry)
```

+++

itertools.islice() allows you to process only a portion of the data stream at a time, without the need to load the entire dataset. This approach enables efficient data processing.


```python
import itertools

# Lazily read file lines with a generator
large_log = (log_entry for log_entry in open("large_log_file.log"))

# Get the first 100 entries from the generator
for entry in itertools.islice(large_log, 100):
    process_log_entry(entry)
```
