---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: venv
  language: python
  name: python3
---

## Alternative Approach

+++

This section covers some alternatives approaches to work with Python. 

+++

### Simplify Null Checks in Python with the Maybe Container

```{code-cell} ipython3
:tags: [hide-cell]

!pip install returns
```

```{code-cell} ipython3
:tags: [hide-cell]

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

```{code-cell} ipython3
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

```{code-cell} ipython3
calculate_discounted_price()
```

The `Maybe` container from the `returns` library enhances code clarity through the `bind_optional` method, which applies a function to the result of the previous step only when that result is not None.

```{code-cell} ipython3
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

```{code-cell} ipython3
calculate_discounted_price()
```

[Link to returns](https://bit.ly/3vUFdGW).

+++

### Box: Using Dot Notation to Access Keys in a Python Dictionary

```{code-cell} ipython3
:tags: [hide-cell]

!pip install python-box[all]
```

Do you wish to use `dict.key` instead of `dict['key']` to access the values inside a Python dictionary? If so, try Box.

Box is like a Python dictionary except that it allows you to access keys using dot notation. This makes the code cleaner when you want to access a key inside a nested dictionary like below.

```{code-cell} ipython3
from box import Box

food_box = Box({"food": {"fruit": {"name": "apple", "flavor": "sweet"}}})
print(food_box)
```

```{code-cell} ipython3
print(food_box.food.fruit.name)
```

[Link to Box](https://github.com/cdgriffith/Box).

+++

### decorator module: Write Shorter Python Decorators without Nested Functions

```{code-cell} ipython3
:tags: [hide-cell]

!pip install decorator
```

Have you ever wished to write a Python decorator with only one function instead of nested functions like below?

```{code-cell} ipython3
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

If so, try decorator. In the code below, `time_func_simple` produces the exact same results as `time_func_complex`, but `time_func_simple` is easier and short to write.

```{code-cell} ipython3
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

[Check out other things the decorator library can do](https://github.com/micheles/decorator).

+++

### Pipe: A Elegant Alternative to Nested map and filter Calls in Python

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pipe
```

Pipe is a Python library that enables infix notation (pipes), offering a cleaner alternative to nested function calls. Here are some of the most useful methods from the Pipe library:

1. `select` and `where` (aliases for `map` and `filter`):

Python's built-in `map` and `filter` functions are powerful tools for working with iterables, allowing for efficient data transformation and filtering. However, when used together, they can lead to code that's difficult to read due to nested function calls. For example:

```{code-cell} ipython3
nums = [1, 2, 3, 4, 5, 6]

list(
    filter(lambda x: x % 2 == 0, 
           map(lambda x: x ** 2, nums)
    )
)
```

Pipe allows for a more intuitive and readable way of chaining operations:

```{code-cell} ipython3
from pipe import select, where

list(
    nums
    | select(lambda x: x ** 2)
    | where(lambda x: x % 2 == 0)
)
```

In this version, the operations are read from left to right, mirroring the order in which they're applied. The `select` method corresponds to `map`, while `where` corresponds to `filter`. This syntax not only improves readability but also makes it easier to add, remove, or reorder operations in your data processing pipeline.

+++

2. `traverse`:

The `traverse` method recursively unfolds nested iterables, which is useful for flattening deeply nested lists:

```{code-cell} ipython3
from pipe import traverse
```

```{code-cell} ipython3
from pipe import traverse

nested = [[1, 2, [3]], [4, 5]]
flattened = list(nested | traverse)
print(flattened) 
```

3. `chain`:

The `chain` method combines multiple iterables:

```{code-cell} ipython3
from pipe import chain

result = list([[1, 2], [3, 4], [5]] | chain)
print(result)
```

4. `take` and `skip`:

These methods allow you to select or skip a specific number of elements from an iterable:

```{code-cell} ipython3
from pipe import take, skip
from itertools import count

first_five = list(count() | take(5))
print(first_five) 
```

```{code-cell} ipython3
skip_first_two = list([1, 2, 3, 4, 5] | skip(2))
print(skip_first_two) 
```

[Link to pipe](https://github.com/JulienPalard/Pipe).

+++

### PRegEx: Write Human-Readable Regular Expressions

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pregex
```

RegEx is useful for extracting words with matching patterns. However, it can be difficult to read and create. PregEx allows you to write a more human-readable RegEx. 

In the code below, I use PregEx to extract URLs from text. 

```{code-cell} ipython3
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

```{code-cell} ipython3
pre.get_matches(text)  
```

[Full article about PregEx](https://towardsdatascience.com/pregex-write-human-readable-regular-expressions-in-python-9c87d1b1335).

[Link to PregEx](https://github.com/manoss96/pregex).

+++

### parse: Extract Strings Using Brackets

```{code-cell} ipython3
:tags: [hide-cell]

!pip install parse
```

If you want to extract substrings from a string, but find it challenging to do so with RegEx, try parse. parse makes it easy to extract strings that are inside brackets. 

```{code-cell} ipython3
from parse import parse 

# Get strings in the brackets
parse("I'll get some {} from {}", "I'll get some apples from Aldi")
```

You can also make the brackets more readable by adding the field name to them.

```{code-cell} ipython3
# Specify the field names for the brackets
parse("I'll get some {items} from {store}", "I'll get some shirts from Walmart")
```

parse also allows you to get the string with a certain format.

```{code-cell} ipython3
# Get a digit and a word
r = parse("I saw {number:d} {animal:w}s", "I saw 3 deers")
r
```

```{code-cell} ipython3
r['number']
```

[Link to parse](https://github.com/r1chardj0n3s/parse).

+++

### Simplify Pattern Matching and Transformation in Python with Pampy

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pampy
```

To simplify extracting and modifying complex Python objects, use Pampy. Pampy enables pattern matching across a variety of Python objects, including lists, dictionaries, tuples, and classes.

```{code-cell} ipython3
from pampy import match, HEAD, TAIL, _

nums = [1, 2, 3]
match(nums, [1, 2, _], lambda num: f"It's {num}")
```

```{code-cell} ipython3
match(nums, [1, TAIL], lambda t: t)
```

```{code-cell} ipython3
nums = [1, [2, 3], 4]

match(nums, [1, [_, 3], _], lambda a, b: [1, a, 3, b])
```

```{code-cell} ipython3
pet = {"type": "dog", "details": {"age": 3}}

match(pet, {"details": {"age": _}}, lambda age: age)
```

[Link to Pampy](https://github.com/santinic/pampy).

+++

### Dictdiffer: Find the Differences Between Two Dictionaries

```{code-cell} ipython3
:tags: [hide-cell]

!pip install dictdiffer
```

When comparing two complicated dictionaries, it is useful to have a tool that finds the differences between the two. Dictdiffer allows you to do exactly that. 

```{code-cell} ipython3
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

```{code-cell} ipython3
# find the difference between two dictionaries
result = diff(user1, user2)
list(result)
```

```{code-cell} ipython3
# swap the diff result
result = diff(user1, user2)
swapped = swap(result)
list(swapped)
```

[Link to Dictdiffer](https://github.com/inveniosoftware/dictdiffer).

+++

### unyt: Manipulate and Convert Units in NumPy Arrays

```{code-cell} ipython3
:tags: [hide-cell]

!pip install unyt 
```

Working with NumPy arrays that have units can be difficult, as it is not immediately clear what the units are, which can lead to errors. 

The unyt package solves this by providing a subclass of NumPy's ndarray class that knows units.

```{code-cell} ipython3
import numpy as np

temps = np.array([25, 30, 35, 40])

temps_f = (temps * 9/5) + 32
print(temps_f)
```

```{code-cell} ipython3
from unyt import degC, degF

# Create an array of temperatures in Celsius
temps = np.array([25, 30, 35, 40]) * degC

# Convert the temperatures to Fahrenheit
temps_f = temps.to(degF)
print(temps_f)
```

unyt arrays support standard NumPy array operations and functions while also preserving the units associated with the data.

```{code-cell} ipython3
temps_f.reshape(2, 2)
```

[Link to unyt](https://github.com/yt-project/unyt).

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Using natsort for Intuitive Alphanumeric Sorting in Python

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
!pip install 'natsort[fast]'
```

When sorting a list of strings containing numbers, Python's default sorting algorithm operates lexicographically. This can lead to unexpected results, especially when dealing with measurements or alphanumeric data:

```{code-cell} ipython3
a = ['2 ft 7 in', '1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '7 ft 6 in']
sorted(a)
```

As you can see, the default `sorted()` function produces a result that doesn't align with our intuitive understanding of numerical order. It places '10 ft 2 in' before '2 ft 11 in' because it compares the strings character by character.

The natsort library solves this problem by providing natural sorting functionality that handles numbers within strings intelligently.

```{code-cell} ipython3
from natsort import natsorted

a = ['2 ft 7 in', '1 ft 5 in', '10 ft 2 in', '2 ft 11 in', '7 ft 6 in']
natsorted(a)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

This makes natsort particularly useful when dealing with alphanumeric data, such as filenames, version numbers, or measurements.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

[Link to natsort](https://github.com/SethMMorton/natsort).

+++

### smart_open: The Python Library That Makes Cloud Storage Feel Local

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
pip install "smart_open[s3]"
```

Working with large remote files in cloud storage services such as S3 often involves complex boilerplate code and careful management of file-like objects, which can lead to subtle bugs.

Let's first look at how we typically interact with S3 using boto3, the AWS SDK for Python:

```{code-cell} ipython3
import boto3

# Initialize S3 client
s3_client = boto3.client('s3')

with open('example_file.txt', 'w') as local_file:
    local_file.write("Hello, world!")

s3_client.upload_file('example_file.txt', 'khuyen-bucket', 'remote_file.txt')
s3_client.download_file('khuyen-bucket', 'remote_file.txt', 'example_file2.txt')

with open('example_file2.txt', 'r') as local_file:
    content = local_file.read()
    print(content)
```

As you can see, this approach requires initializing an S3 client, managing file-like objects, and using separate methods for uploading and downloading. It's not particularly intuitive, especially for developers who are used to working with local files.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

smart_open addresses these issues by providing a single `open()` function that works across different storage systems and file formats. Let's see how it simplifies our S3 operations:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from smart_open import open

with open('s3://khuyen-bucket/example_file.txt', 'w') as s3_file:
    s3_file.write("Hello, world!")


with open('s3://khuyen-bucket/example_file.txt', 'r') as s3_file:
    print(s3_file.read())
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Another great feature of smart_open is its ability to handle compressed files transparently. Let's say we have a gzipped file that we want to upload to S3 and then read from:

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
!gzip example_file.txt
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# Uploading a gzipped file
with open('example_file.txt.gz', 'r') as local_file:
    with open('s3://khuyen-bucket/example_file.txt.gz', 'w') as s3_file:
        s3_file.write(local_file.read())
        
# Reading a gzipped file from S3
with open('s3://khuyen-bucket/example_file.txt.gz', 'r') as s3_file:
    content = s3_file.read()
    print(content)
```

[Link to smart_open](https://github.com/piskvorky/smart_open).

+++

### Flicking: Safe Model Deserialization in Python

```{code-cell} ipython3
!pip install fickling
```

When working with machine learning models or serialized data in Python, it's common to use the pickle module to save and load data. However, loading pickle files directly without security checks can result in potential code execution vulnerabilities, especially when handling untrusted ML models or serialized data from external sources.

+++

Let's consider an example where we create a simple dummy model class, save it to a pickle file, and then load it without any safety checks.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import pickle
import numpy as np


# Create a simple dummy model class
class DummyModel:
    def __init__(self):
        self.weights = np.random.rand(10)

    def predict(self, X):
        return np.dot(X, self.weights)


# Create an instance of the dummy model
model = DummyModel()

# Save the model to a pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

```{code-cell} ipython3
import pickle


# No safety checks, potential for malicious code execution
with open("model.pkl", "rb") as f:
    data = pickle.load(f)  # Could execute harmful code
```

As you can see, loading the pickle file without any safety checks can potentially execute malicious code, which is a serious security risk.

+++

Fickling provides several ways to safely handle pickle files and ML models by detecting malicious content before execution. You can:

- Add runtime safety checks for all pickle operations
- Get detailed analysis of potential security issues

Here's an example of how to use Fickling to safely load a pickle file:

```{code-cell} ipython3
import fickling

fickling.always_check_safety()

with open("model.pkl", "rb") as f:
    data = pickle.load(f)
```

When we run this code, Fickling will raise an `UnsafeFileError` if it detects any potential security issues.

+++

[Link to Flicking](https://github.com/trailofbits/fickling).

+++

### Safe Unit Conversions in Python Using Pint

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
!pip install pint
```

Working with physical quantities and unit conversions in Python leads to error-prone calculations and confusing code, as units are often stored separately or ignored altogether, which results in hard-to-catch bugs and inconsistent results.

Let's consider an example where we're calculating speed from distance and time. Without explicit unit definitions, it's unclear what units are being used, making it easy for developers to make incorrect assumptions.

```{code-cell} ipython3
distance = 100  # meters? feet? kilometers?
time = 9.58     # seconds? minutes?

# Manual conversion needed
# assumes distance was in kilometers while in fact it was in meters
speed_kmh = (distance) / (time / 3600)
```

With Pint, you can define units explicitly, and it will handle unit tracking and conversions automatically.


```{code-cell} ipython3
import pint

ureg = pint.UnitRegistry()

# Clear unit definitions
distance = 100 * ureg.meters
time = 10 * ureg.seconds

# Automatic unit conversion
speed = distance / time
speed
```

```{code-cell} ipython3
speed.to("kilometers/hour")
```

Pint also ensures dimensional consistency in calculations, raising an error if units don't match.

```{code-cell} ipython3
# Will raise error if units don't match
try:
    wrong_calc = 3 * ureg.meters + 4 * ureg.seconds
except pint.DimensionalityError as e:
    print("DimensionalityError:", e)
```

[Link to Pint](https://github.com/hgrecco/pint).
