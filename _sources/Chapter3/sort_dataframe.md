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

## Sort Rows or Columns of a DataFrame

+++

### set_categories in Pandas: Sort Categorical Column by a Specific Ordering

+++

If you want to sort pandas DataFrame’s categorical column by a specific ordering such as small, medium, large, use `df.col.cat.set_categories()` method.

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame(
    {"col1": ["large", "small", "mini", "medium", "mini"], "col2": [1, 2, 3, 4, 5]}
)
ordered_sizes = "large", "medium", "small", "mini"

df.col1 = df.col1.astype("category")
df.col1.cat.set_categories(ordered_sizes, ordered=True, inplace=True)
df.sort_values(by="col1")
```
