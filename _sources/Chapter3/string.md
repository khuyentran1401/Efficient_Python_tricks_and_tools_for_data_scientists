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

## Work with String

+++

### pandas.Series.str: Manipulate Text Data in a Pandas Series	

+++

If you are working the text data in a pandas Series, instead of creating your own functions, use `pandas.Series.str` to access common methods to process string.

The code below shows how to convert text to lower case then replace “e” with “a”.

```{code-cell} ipython3
import pandas as pd 

fruits = pd.Series(['Orange', 'Apple', 'Grape'])
fruits
```

```{code-cell} ipython3
fruits.str.lower()
```

```{code-cell} ipython3
fruits.str.lower().str.replace("e", "a")
```

Find other useful string methods [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#string-methods).

+++

### DataFrame.columns.str.startswith: Find DataFrame’s Columns that Start With a Pattern

+++

To find pandas DataFrame whose columns start with a pattern, use `df.columns.str.startswith`. 

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({'pricel': [1, 2, 3],
                    'price2': [2, 3, 4],
                    'year': [2020, 2021, 2021]})

mask = df.columns.str.startswith('price')
df.loc[:, mask]
```

### Find Rows Containing One of the Substrings in a List

+++

If you want to find rows that contain one of the substrings in a list, join that list using `|`:

```{code-cell} ipython3
import pandas as pd  

s = pd.Series(['bunny', 'monkey', 'funny', 'flower'])

sub_str = ['ny', 'ey']
join_str = '|'.join(sub_str)
join_str
```

... then use `str.contains`. Now you only get the strings that end with "ny" or "ey":

```{code-cell} ipython3
s[s.str.contains(join_str)]
```
