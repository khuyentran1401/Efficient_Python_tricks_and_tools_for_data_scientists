## Alternative Approach

This section covers some alternatives approaches to work with Python. 

### Simplify Null Checks in Python with the Maybe Container


```python
!pip install returns
```


```python
from typing import Optional


class Event:
    def __init__(self, ticket: Ticket) -> None:
        self._ticket = ticket

    def get_ticket(self) -> Ticket:
        return self._ticket


class Ticket:
    def __init__(self, price: float) -> None:
        self._price = price

    def get_price(self) -> float:
        return self._price


class Discount:
    def __init__(self, discount_amount: float):
        self.discount_amount = discount_amount

    def apply_discount(self, price: float) -> float:
        return price - self.discount_amount
```

Having multiple `if x is not None:` conditions can make the code deeply nested and unreadable.


```python
def calculate_discounted_price(
    event: Optional[Event] = None, discount: Optional[Discount] = None
) -> Optional[float]:
    if event is not None:
        ticket = event.get_ticket()
        if ticket is not None:
            price = ticket.get_price()
            if discount is not None:
                return discount.apply_discount(price)
    return None


ticket = Ticket(100)
concert = Event(ticket)
discount = Discount(20)
calculate_discounted_price(concert, discount)
```




    80




```python
calculate_discounted_price()
```

The `Maybe` container from the `returns` library enhances code clarity through the `bind_optional` method, which applies a function to the result of the previous step only when that result is not None.


```python
from returns.maybe import Maybe


def calculate_discounted_price(
    event: Optional[Event] = None, discount: Optional[Discount] = None
) -> Maybe[float]:
    return (
        Maybe.from_optional(event)
        .bind_optional(lambda event: event.get_ticket()) # called only when event exists
        .bind_optional(lambda ticket: ticket.get_price()) # called only when ticket exists
        .bind_optional(lambda price: discount.apply_discount(price)) # called only when price exists
    )

ticket = Ticket(100)
concert = Event(ticket)
discount = Discount(20)
calculate_discounted_price(concert, discount)
```




    <Some: 80>




```python
calculate_discounted_price()
```




    <Nothing>



[Link to returns](https://bit.ly/3vUFdGW).

### Box: Using Dot Notation to Access Keys in a Python Dictionary



```python
!pip install python-box[all]
```

Do you wish to use `dict.key` instead of `dict['key']` to access the values inside a Python dictionary? If so, try Box.

Box is like a Python dictionary except that it allows you to access keys using dot notation. This makes the code cleaner when you want to access a key inside a nested dictionary like below.


```python
from box import Box

food_box = Box({"food": {"fruit": {"name": "apple", "flavor": "sweet"}}})
print(food_box)
```

    {'food': {'fruit': {'name': 'apple', 'flavor': 'sweet'}}}



    <IPython.core.display.Javascript object>



```python
print(food_box.food.fruit.name)
```

    apple



    <IPython.core.display.Javascript object>


[Link to Box](https://github.com/cdgriffith/Box).

### decorator module: Write Shorter Python Decorators without Nested Functions


```python
!pip install decorator
```

Have you ever wished to write a Python decorator with only one function instead of nested functions like below?



```python
from time import time, sleep


def time_func_complex(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        func(*args, **kwargs)
        end_time = time()
        print(
            f"""It takes {round(end_time - start_time, 3)} seconds to execute the function"""
        )

    return wrapper


@time_func_complex
def test_func_complex():
    sleep(1)


test_func_complex()
```

    It takes 1.001 seconds to execute the function



    <IPython.core.display.Javascript object>


If so, try decorator. In the code below, `time_func_simple` produces the exact same results as `time_func_complex`, but `time_func_simple` is easier and short to write.



```python
from decorator import decorator


@decorator
def time_func_simple(func, *args, **kwargs):
    start_time = time()
    func(*args, **kwargs)
    end_time = time()
    print(
        f"""It takes {round(end_time - start_time, 3)} seconds to execute the function"""
    )


@time_func_simple
def test_func_simple():
    sleep(1)


test_func_simple()
```

    It takes 1.001 seconds to execute the function



    <IPython.core.display.Javascript object>


[Check out other things the decorator library can do](https://github.com/micheles/decorator).

### Pipe: A Elegant Alternative to Nested map and filter Calls in Python


```python
!pip install pipe
```

Pipe is a Python library that enables infix notation (pipes), offering a cleaner alternative to nested function calls. Here are some of the most useful methods from the Pipe library:

1. `select` and `where` (aliases for `map` and `filter`):

Python's built-in `map` and `filter` functions are powerful tools for working with iterables, allowing for efficient data transformation and filtering. However, when used together, they can lead to code that's difficult to read due to nested function calls. For example:


```python
nums = [1, 2, 3, 4, 5, 6]

list(
    filter(lambda x: x % 2 == 0, 
           map(lambda x: x ** 2, nums)
    )
)
```




    [4, 16, 36]



Pipe allows for a more intuitive and readable way of chaining operations:


```python
from pipe import select, where

list(
    nums
    | select(lambda x: x ** 2)
    | where(lambda x: x % 2 == 0)
)
```




    [4, 16, 36]



In this version, the operations are read from left to right, mirroring the order in which they're applied. The `select` method corresponds to `map`, while `where` corresponds to `filter`. This syntax not only improves readability but also makes it easier to add, remove, or reorder operations in your data processing pipeline.

2. `traverse`:

The `traverse` method recursively unfolds nested iterables, which is useful for flattening deeply nested lists:


```python
from pipe import traverse

```


```python
from pipe import traverse

nested = [[1, 2, [3]], [4, 5]]
flattened = list(nested | traverse)
print(flattened) 
```

    [1, 2, 3, 4, 5]


3. `chain`:

The `chain` method combines multiple iterables:


```python
from pipe import chain

result = list([[1, 2], [3, 4], [5]] | chain)
print(result)
```

    [1, 2, 3, 4, 5]


4. `take` and `skip`:

These methods allow you to select or skip a specific number of elements from an iterable:


```python
from pipe import take, skip
from itertools import count

first_five = list(count() | take(5))
print(first_five) 
```

    [0, 1, 2, 3, 4]



```python
skip_first_two = list([1, 2, 3, 4, 5] | skip(2))
print(skip_first_two) 
```

    [3, 4, 5]


[Link to pipe](https://bit.ly/4d90FIJ).

### PRegEx: Write Human-Readable Regular Expressions


```python
!pip install pregex
```

RegEx is useful for extracting words with matching patterns. However, it can be difficult to read and create. PregEx allows you to write a more human-readable RegEx. 

In the code below, I use PregEx to extract URLs from text. 


```python
from pregex.core.classes import AnyButWhitespace
from pregex.core.quantifiers import OneOrMore, Optional
from pregex.core.operators import Either


text = "You can find me through my website mathdatasimplified.com/ or GitHub https://github.com/khuyentran1401"

any_but_space = OneOrMore(AnyButWhitespace())
optional_scheme = Optional("https://")
domain = Either(".com", ".org")

pre = (
    optional_scheme
    + any_but_space
    + domain
    + any_but_space
)

pre.get_pattern()

```




    '(?:https:\\/\\/)?\\S+(?:\\.com|\\.org)\\S+'




```python
pre.get_matches(text)  
```




    ['mathdatasimplified.com/', 'https://github.com/khuyentran1401']



[Full article about PregEx](https://towardsdatascience.com/pregex-write-human-readable-regular-expressions-in-python-9c87d1b1335).

[Link to PregEx](https://github.com/manoss96/pregex).

### parse: Extract Strings Using Brackets


```python
!pip install parse
```

If you want to extract substrings from a string, but find it challenging to do so with RegEx, try parse. parse makes it easy to extract strings that are inside brackets. 


```python
from parse import parse 

# Get strings in the brackets
parse("I'll get some {} from {}", "I'll get some apples from Aldi")
```




    <Result ('apples', 'Aldi') {}>



You can also make the brackets more readable by adding the field name to them.


```python
# Specify the field names for the brackets
parse("I'll get some {items} from {store}", "I'll get some shirts from Walmart")
```




    <Result () {'items': 'shirts', 'store': 'Walmart'}>



parse also allows you to get the string with a certain format.


```python
# Get a digit and a word
r = parse("I saw {number:d} {animal:w}s", "I saw 3 deers")
r
```




    <Result () {'number': 3, 'animal': 'deer'}>




```python
r['number']
```




    3



[Link to parse](https://github.com/r1chardj0n3s/parse).

### Simplify Pattern Matching and Transformation in Python with Pampy


```python
!pip install pampy
```

To simplify extracting and modifying complex Python objects, use Pampy. Pampy enables pattern matching across a variety of Python objects, including lists, dictionaries, tuples, and classes.


```python
from pampy import match, HEAD, TAIL, _

nums = [1, 2, 3]
match(nums, [1, 2, _], lambda num: f"It's {num}")

```




    "It's 3"




```python
match(nums, [1, TAIL], lambda t: t)

```




    [2, 3]




```python
nums = [1, [2, 3], 4]

match(nums, [1, [_, 3], _], lambda a, b: [1, a, 3, b])

```




    [1, 2, 3, 4]




```python
pet = {"type": "dog", "details": {"age": 3}}

match(pet, {"details": {"age": _}}, lambda age: age)

```




    3



[Link to Pampy](https://github.com/santinic/pampy).

### Dictdiffer: Find the Differences Between Two Dictionaries


```python
!pip install dictdiffer
```

When comparing two complicated dictionaries, it is useful to have a tool that finds the differences between the two. Dictdiffer allows you to do exactly that. 


```python
from dictdiffer import diff, swap

user1 = {
    "name": "Ben", 
    "age": 25, 
    "fav_foods": ["ice cream"],
}

user2 = {
    "name": "Josh",
    "age": 25,
    "fav_foods": ["ice cream", "chicken"],
}

```


```python
# find the difference between two dictionaries
result = diff(user1, user2)
list(result)
```




    [('change', 'name', ('Ben', 'Josh')), ('add', 'fav_foods', [(1, 'chicken')])]




    <IPython.core.display.Javascript object>



```python
# swap the diff result
result = diff(user1, user2)
swapped = swap(result)
list(swapped)
```




    [('change', 'name', ('Josh', 'Ben')),
     ('remove', 'fav_foods', [(1, 'chicken')])]




    <IPython.core.display.Javascript object>


[Link to Dictdiffer](https://github.com/inveniosoftware/dictdiffer).

### unyt: Manipulate and Convert Units in NumPy Arrays


```python
!pip install unyt 
```

Working with NumPy arrays that have units can be difficult, as it is not immediately clear what the units are, which can lead to errors. 

The unyt package solves this by providing a subclass of NumPy's ndarray class that knows units.


```python
import numpy as np

temps = np.array([25, 30, 35, 40])

temps_f = (temps * 9/5) + 32
print(temps_f)
```

    [ 77.  86.  95. 104.]



```python
from unyt import degC, degF

# Create an array of temperatures in Celsius
temps = np.array([25, 30, 35, 40]) * degC

# Convert the temperatures to Fahrenheit
temps_f = temps.to(degF)
print(temps_f)
```

    [ 77.  86.  95. 104.] °F


unyt arrays support standard NumPy array operations and functions while also preserving the units associated with the data.


```python
temps_f.reshape(2, 2)
```




    unyt_array([[ 77., 572.],
                [ 95., 104.]], 'degF')



[Link to unyt](https://github.com/yt-project/unyt).

### Map a Function Asynchronously with Prefect


```python
!pip install -U prefect 
```

`map` runs a function for each item in an iterable synchronously.


```python
def add_one(x):
    sleep(2)
    return x + 1

def sync_map():
    b = [add_one(item) for item in [1, 2, 3]]

sync_map()
```

![](../img/sync_map.png)

To speed up the execution, map a function asynchronously with Prefect. 


```python
from prefect import flow, task
from time import sleep
import warnings

warnings.simplefilter("ignore", UserWarning)

# Create a task
@task
def add_one(x):
    sleep(2)
    return x + 1

# Create a flow
@flow
def async_map():
    # Run a task for each element in the iterable
    b = add_one.map([1, 2, 3])


async_map()

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:49.018 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | prefect.engine - Created flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> for flow<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> 'async-map'</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:51.144 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Created task run 'add_one-3c3112ef-1' for task 'add_one'
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:51.148 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Submitted task run 'add_one-3c3112ef-1' for execution.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:51.191 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Created task run 'add_one-3c3112ef-0' for task 'add_one'
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:51.192 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Submitted task run 'add_one-3c3112ef-0' for execution.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:51.208 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Created task run 'add_one-3c3112ef-2' for task 'add_one'
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:51.210 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Submitted task run 'add_one-3c3112ef-2' for execution.
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:54.321 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Task run 'add_one-3c3112ef-0' - Finished in state <span style="color: #008000; text-decoration-color: #008000">Completed</span>()
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:54.362 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Task run 'add_one-3c3112ef-2' - Finished in state <span style="color: #008000; text-decoration-color: #008000">Completed</span>()
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:54.380 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Task run 'add_one-3c3112ef-1' - Finished in state <span style="color: #008000; text-decoration-color: #008000">Completed</span>()
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">09:04:54.685 | <span style="color: #008080; text-decoration-color: #008080">INFO</span>    | Flow run<span style="color: #800080; text-decoration-color: #800080"> 'copper-sponge'</span> - Finished in state <span style="color: #008000; text-decoration-color: #008000">Completed</span>('All states completed.')
</pre>




    [Completed(message=None, type=COMPLETED, result=PersistedResult(type='reference', serializer_type='pickle', storage_block_id=UUID('45e1a1fc-bdc8-4f8d-8945-287d12b46d33'), storage_key='ad7912161ab44a6d8359f8089a16202d')),
     Completed(message=None, type=COMPLETED, result=PersistedResult(type='reference', serializer_type='pickle', storage_block_id=UUID('45e1a1fc-bdc8-4f8d-8945-287d12b46d33'), storage_key='fe83574cd0df4fc5838ef902beb34f6b')),
     Completed(message=None, type=COMPLETED, result=PersistedResult(type='reference', serializer_type='pickle', storage_block_id=UUID('45e1a1fc-bdc8-4f8d-8945-287d12b46d33'), storage_key='ba18fe9c568845ecbad03c25df353655'))]


![](../img/async_map.png)

Prefect is an open-source library that allows you to orchestrate and observe your data pipelines defined in Python. Check out the [getting started tutorials](https://docs.prefect.io/tutorials/first-steps/) for basic concepts of Prefect. 
