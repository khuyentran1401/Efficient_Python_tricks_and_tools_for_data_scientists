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

## Test

+++

This section shows how to compare between 2 Pandas DataFrame or between 2 Pandas Series

+++

### assert_frame equal: Test Whether Two DataFrames are Similar

+++

If you want to test whether two DataFrames are similar or how much they are different from each other, try `pandas.testing.assert_frame_equal`.

```{code-cell} ipython3
from pandas.testing import assert_frame_equal
import pandas as pd


df1 = pd.DataFrame({"coll": [1, 2, 3], "col2": [4, 5, 6]})
df2 = pd.DataFrame({"coll": [1, 3, 4], "col2": [4, 5, 6]})
assert_frame_equal(df1, df2)
```

### Ignore the Order of Index When Comparing Two DataFrames 

+++

If you want to ignore the order of index & columns when comparing two DataFrames , use `assert_frame_equal(df1, df2, check_like=True)`.

```{code-cell} ipython3
from pandas.testing import assert_frame_equal
import pandas as pd


df1 = pd.DataFrame({"coll": [1, 2, 3], "col2": [4, 5, 6]})
df2 = pd.DataFrame({"col2": [4, 5, 6], "coll": [1, 2, 3]})
assert_frame_equal(df1, df2, check_like=True)
```

```{code-cell} ipython3
df1 = pd.DataFrame({"coll": [1, 2, 3], "col2": [4, 5, 6]})
df2 = pd.DataFrame({"col2": [4, 5, 6], "coll": [1, 2, 3]})
assert_frame_equal(df1, df2)
```

### Compare the Difference Between Two DataFrames

+++

If you want to show and align the differences between two DataFrames, use `df.compare`.

```{code-cell} ipython3
import pandas as pd

df1 = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
df2 = pd.DataFrame({"col1": [1, 3, 4], "col2": [4, 5, 6]})

df1.compare(df2)
```
