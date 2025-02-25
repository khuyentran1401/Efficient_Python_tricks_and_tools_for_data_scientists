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

## Transform a DataFrame

+++

This section covers some pandas methods to transform a DataFrame into another form using methods such as aggregation and groupby.

+++

### pandas.DataFrame.agg: Aggregate over Columns or Rows Using Multiple Operations

+++

If you want to aggregate over columns or rows using one or more operations, try `pd.DataFrame.agg`.

```{code-cell} ipython3
from collections import Counter
import pandas as pd


def count_two(nums: list):
    return Counter(nums)[2]


df = pd.DataFrame({"coll": [1, 3, 5], "col2": [2, 4, 6]})
df.agg(["sum", count_two])
```

### pandas.DataFrame.agg: Apply Different Aggregations to Different Columns

+++

If you want to apply different aggregations to different columns, insert a dictionary of column and aggregation methods to the `pd.DataFrame.agg` method.

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})

df.agg({"a": ["sum", "mean"], "b": ["min", "max"]})
```

### Group DataFrame's Rows into a List Using groupby

+++

It is common to use `groupby` to get the statistics of rows in the same group such as count, mean, median, etc. If you want to group rows into a list instead, use `lambda x: list(x)`.

```{code-cell} ipython3
import pandas as pd

df = pd.DataFrame(
    {
        "col1": [1, 2, 3, 4, 3],
        "col2": ["a", "a", "b", "b", "c"],
        "col3": ["d", "e", "f", "g", "h"],
    }
)

df.groupby(["col2"]).agg({"col1": "mean", "col3": lambda x: list(x)})
```

### Get the N Largest Values for Each Category in a DataFrame

+++

If you want to get the `n` largest values for each category in a pandas DataFrame, use the combination of `groupby` and `nlargest`. 

```{code-cell} ipython3
import pandas as pd

df = pd.DataFrame({"type": ["a", "a", "a", "b", "b"], "value": [1, 2, 3, 1, 2]})

# Get n largest values for each type
(
    df.groupby("type")
    .apply(lambda df_: df_.nlargest(2, "value"))
    .reset_index(drop=True)
)
```

### Assign Name to a Pandas Aggregation

+++

By default, aggregating a column returns the name of that column.

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({"size": ["S", "S", "M", "L"], "price": [2, 3, 4, 5]})

df.groupby('size').agg({'price': 'mean'})
```

If you want to assign a new name to an aggregation, add `name = (column, agg_method)` to `agg`.

```{code-cell} ipython3
df.groupby('size').agg(mean_price=('price', 'mean'))
```

### pandas.pivot_table: Turn Your DataFrame Into a Pivot Table

+++

A pivot table is useful to summarize and analyze the patterns in your data. If you want to turn your DataFrame into a pivot table, use `pandas.pivot_table`.

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame(
    {
        "item": ["apple", "apple", "apple", "apple", "apple"],
        "size": ["small", "small", "large", "large", "large"],
        "location": ["Walmart", "Aldi", "Walmart", "Aldi", "Aldi"],
        "price": [3, 2, 4, 3, 2.5],
    }
)

df
```

```{code-cell} ipython3
pivot = pd.pivot_table(
    df, values="price", index=["item", "size"], columns=["location"], aggfunc="mean"
)
pivot
```

### DataFrame.groupby.sample: Get a Random Sample of Items from Each Category in a Column	

+++

If you want to get a random sample of items from each category in a column, use `pandas.DataFrame.groupby.sample`.This method is useful when you want to get a subset of a DataFrame while keeping all categories in a column.

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({"col1": ["a", "a", "b", "c", "c", "d"], "col2": [4, 5, 6, 7, 8, 9]})
df.groupby("col1").sample(n=1)
```

To get 2 items from each category, use `n=2`.

```{code-cell} ipython3
df = pd.DataFrame(
    {
        "col1": ["a", "a", "b", "b", "b", "c", "c", "d", "d"],
        "col2": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
)
df.groupby("col1").sample(n=2)
```

### size: Compute the Size of Each Group

+++

If you want to get the count of elements in one column, use `groupby` and `count`. 

```{code-cell} ipython3
import pandas as pd

df = pd.DataFrame(
    {"col1": ["a", "b", "b", "c", "c", "d"], "col2": ["S", "S", "M", "L", "L", "L"]}
)

df.groupby(['col1']).count()
```

If you want to get the size of groups composed of 2 or more columns, use `groupby` and `size` instead. 

```{code-cell} ipython3
df.groupby(['col1', 'col2']).size()
```

### pandas.melt: Unpivot a DataFrame

+++

If you want to unpivot a DataFrame from wide to long format, use pandas.melt.

+++

For example, you can use pandas.melt to turn multiple columns (`Aldi`, `Walmart`, `Costco`) into values of one column (`store`).  

```{code-cell} ipython3
import pandas as pd

df = pd.DataFrame(
    {"fruit": ["apple", "orange"], "Aldi": [1, 2], "Walmart": [3, 4], "Costco": [5, 6]}
)
df
```

```{code-cell} ipython3
df.melt(id_vars=["fruit"], value_vars=["Aldi", "Walmart", "Costco"], var_name='store')
```

### pandas.crosstab: Create a Cross Tabulation

+++

Cross tabulation allows you to analyze the relationship between multiple variables. To turn a pandas DataFrame into a cross tabulation, use `pandas.crosstab`.

```{code-cell} ipython3
import pandas as pd  

network = [
    ("Ben", "Khuyen"),
    ("Ben", "Josh"),
    ("Lauren", "Thinh"),
    ("Lauren", "Khuyen"),
    ("Khuyen", "Josh"),
]

# Create a dataframe of the network
friends1 = pd.DataFrame(network, columns=["person1", "person2"])

# Reverse the order of the columns
friends2 = pd.DataFrame(network, columns=["person2", "person1"])

# Create a symmetric dataframe
friends = pd.concat([friends1, friends2])

# Create a cross tabulation 
pd.crosstab(friends.person1, friends.person2)
```

### Stack Columns into Rows in Pandas

+++

If you want to stack the columns into rows, use `DataFrame.stack()`.

```{code-cell} ipython3
import pandas as pd

# Create a DataFrame with two columns and a MultiIndex
df = pd.DataFrame(
    {"A": [1, 2, 3], "B": [4, 5, 6]}, index=["x", "y", "z"]
)

# Original DataFrame
print("Original DataFrame:")
print(df)
```

```{code-cell} ipython3
# Stacked DataFrame
stacked_df = df.stack()

print("\nStacked DataFrame:")
print(stacked_df)
```

### Turn a pandas Series into a pandas DataFrame

+++

If you want to turn a pandas Series into a pandas DataFrame, use `str.get_dummies()`.

```{code-cell} ipython3
import pandas as pd

s = pd.Series(["a", "b", "a,b", "a,c"])
s 
```

```{code-cell} ipython3
# Split the string by ","
s.str.get_dummies(sep=",")
```

### Align Pandas Objects for Effective Data Manipulation

+++

To perform operations between two pandas objects, it's often necessary to ensure that two pandas objects have the same row or column labels. The `df.align` method allows you to align two pandas objects along specified axes.

```{code-cell} ipython3
import pandas as pd

df1 = pd.DataFrame([[1, 2], [4, 5]], columns=["b", "a"])
df2 = pd.DataFrame([[4, 3, 2], [6, 5, 4]], columns=["a", "b", "c"])

print("df1:\n", df1, "\n")
print("df2:\n", df2)
```

```{code-cell} ipython3
# ensure df1 and df2 have the same column labels
# by including all unique column labels from both objects
left, right = df1.align(df2, join='outer', axis=1)
```

```{code-cell} ipython3
print("df1:\n", left, "\n")
print("df2:\n", right)
```
