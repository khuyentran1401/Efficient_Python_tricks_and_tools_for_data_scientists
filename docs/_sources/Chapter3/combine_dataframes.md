---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: 'Python 3.8.9 (''venv'': venv)'
  language: python
  name: python3
---

## Combine Multiple DataFrames

+++

### pandas.DataFrame.combine_first: Update Null Elements Using Another DataFrame

+++

If you want to fill null values in one DataFrame with non-null values at the same locations from another DataFrame, use `pandas.DataFrame.combine_first`.

```{code-cell} ipython3
import pandas as pd

store1 = pd.DataFrame({"orange": [None, 5], "apple": [None, 1]})
store2 = pd.DataFrame({"orange": [1, 3], "apple": [4, 2]})
```

```{code-cell} ipython3
print(store1)
```

```{code-cell} ipython3
print(store2)
```

In the code below, the values at the first row of `store1` are updated with the values at the first row of `store2`.

```{code-cell} ipython3
print(store1.combine_first(store2))
```

### df.merge: Merge DataFrame Based on Columns

+++

If you want to merge on DataFrame with another DataFrame based on the similarity between 2 columns, use `df.merge`.

+++

If you want to merge on DataFrame with another DataFrame based on the similarity between 2 columns, use `df.merge`.

+++

If you want to merge on DataFrame with another DataFrame based on the similarity between 2 columns, use `df.merge`.

```{code-cell} ipython3
import pandas as pd

df1 = pd.DataFrame({"key1": ["a", "a", "b", "b", "a"], "value": [1, 2, 3, 4, 5]})
df2 = pd.DataFrame({"key2": ["a", "b", "c"], "value": [6, 7, 8]})

df1 
```

```{code-cell} ipython3
df2 
```

```{code-cell} ipython3
df1.merge(df2, left_on="key1", right_on="key2")
```

### Enhancing Readability in DataFrame Merging With Custom Suffixes

+++

When merging two DataFrames with overlapping columns, the default behavior is to add suffixes "_x" and "_y" to the column names.

```{code-cell} ipython3
import pandas as pd

df1 = pd.DataFrame({"id": [1, 2, 3], "val": ["A", "B", "C"]})
df2 = pd.DataFrame({"id": [2, 3, 4], "val": ["X", "Y", "Z"]})

merged_df = pd.merge(df1, df2, on="id")

print(merged_df)
```

To improve readability, you can specify custom suffixes.

```{code-cell} ipython3
merged_df = pd.merge(df1, df2, on="id", suffixes=("_df1", "_df2"))
print(merged_df)
```

### Include All Rows When Merging Two DataFrames

+++

`df.merge` only includes rows with matching values in both DataFrames. If you want to include all rows from both DataFrames, use `how='outer'`.  

```{code-cell} ipython3
import pandas as pd

df1 = pd.DataFrame({'key': ['A', 'B'], 'v1': [1, 2]})
df2 = pd.DataFrame({'key': ['B', 'C'], 'v2': [3, 4]})

# Keep only rows with matching keys
pd.merge(df1, df2, on='key')
```

```{code-cell} ipython3
# Keep all rows
pd.merge(df1, df2, on='key', how='outer')
```

### DataFrame.insert: Insert a Column Into a DataFrame at a Specified Location

+++

If you want to insert a column into a DataFrame at a specified location, use `df.insert`. In the code below, I insert column c at location 0. 

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
df.insert(loc=0, column='c', value=[5, 6])
```

```{code-cell} ipython3
df 
```
