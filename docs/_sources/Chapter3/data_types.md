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

## Manipulate a DataFrame Using Data Types

+++

### select_dtypes: Return a Subset of a DataFrame Including/Excluding Columns Based on Their dtype

+++

You might want to apply different kinds of processing to categorical and numerical features. Instead of manually choosing categorical features or numerical features, you can automatically get them by using `df.select_dtypes('data_type')`.

In the example below, you can either include or exclude certain data types using `exclude`.


```{code-cell} ipython3
import pandas as pd 
```

```{code-cell} ipython3
df = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3], "col3": [0.1, 0.2, 0.3]})

df.info()
```

```{code-cell} ipython3
df.select_dtypes(include=["int64", "float64"])
```

### Smart Data Type Selection for Memory-Efficient Pandas

+++

To reduce the memory usage of a Pandas DataFrame, you can start by changing the data type of a column. 

```{code-cell} ipython3
from sklearn.datasets import load_iris
import pandas as pd 

X, y = load_iris(as_frame=True, return_X_y=True)
df = pd.concat([X, pd.DataFrame(y, columns=["target"])], axis=1)
df.info()
```

By default, Pandas uses `float64` for floating-point numbers, which can be oversized for columns with smaller value ranges. Here are some alternatives:

- **float16**: Suitable for values between -32768 and 32767.
- **float32**: Suitable for integers between -2147483648 and 2147483647.
- **float64**: The default, suitable for a wide range of values.

For example, if you know that the values in the "sepal length (cm)" column will never exceed 32767, you can use `float16` to reduce memory usage.

```{code-cell} ipython3
df['sepal length (cm)'].max()
```

```{code-cell} ipython3
df['sepal length (cm)'].memory_usage()
```

```{code-cell} ipython3
df["sepal length (cm)"] = df["sepal length (cm)"].astype("float16")
df['sepal length (cm)'].memory_usage()
```

Here, the memory usage of the "sepal length (cm)" column decreased from 1332 bytes to 432 bytes, a reduction of approximately 67.6%.

+++

If you have a categorical variable with low cardinality, you can change its data type to `category` to reduce memory usage.

The "target" column has only 3 unique values, making it a good candidate for the category data type to save memory.

```{code-cell} ipython3
# View category
df['target'].nunique()
```

```{code-cell} ipython3
df['target'].memory_usage()
```

```{code-cell} ipython3
df["target"] = df["target"].astype("category")
df['target'].memory_usage()
```

Here, the memory usage of the "target" column decreased from 1332 bytes to 414 bytes, a reduction of approximately 68.9%.

+++

If we apply this reduction to the rest of the columns, the memory usage of the DataFrame decreased from 6.0 KB to 1.6 KB, a reduction of approximately 73.3%.

```{code-cell} ipython3
float_cols = df.select_dtypes(include=['float64']).columns
df[float_cols] = df[float_cols].apply(lambda x: x.astype('float16'))
df.info()
```

### pandas.Categorical: Turn a List of Strings into a Categorical Variable

+++

If you want to create a categorical variable, use `pandas.Categorical`. This variable takes on a limited number of possible values and can be ordered. In the code below, I use `pd.Categorical` to create a list of ordered categories.

```{code-cell} ipython3
import pandas as pd 

size = pd.Categorical(['M', 'S', 'M', 'L'], ordered=True, categories=['S', 'M', 'L'])
size
```

Note that the parameters `categories = ['S', 'M', 'L']` and `ordered=True` tell pandas that `'S' < 'M' < 'L'`. This means we can get the smallest value in the list:

```{code-cell} ipython3
size.min()
```

Or sort the DataFrame by the column that contains categorical variables:

```{code-cell} ipython3
df = pd.DataFrame({'size': size, 'val': [5, 4, 3, 6]})

df.sort_values(by='size')
```

### Optimizing Memory Usage in a pandas DataFrame with infer_objects

+++

pandas DataFrames that contain columns of mixed data types are stored in a more general format (such as "object"), resulting in inefficient memory usage and slower computation times.

`df.infer_objects()` can infer the true data types of columns in a DataFrame, which can help optimize memory usage in your code.

In the following code, the column "col1" still has an "object" data type even though it contains integer values after removing the first row. 

By using the `df.infer_objects()` method, "col1" is converted to an "int64" data type which saves approximately 27 MB of memory.

```{code-cell} ipython3
import pandas as pd
from random import randint 

random_numbers = [randint(0, 100) for _ in range(1000000)]
df = pd.DataFrame({"col1": ['a', *random_numbers]})

# Remove the first row
df = df.iloc[1:]

print(df.dtypes)
print(df.memory_usage(deep=True))
```

```{code-cell} ipython3
inferred_df = df.infer_objects()
print(inferred_df.dtypes)
print(inferred_df.memory_usage(deep=True))
```

### Say Goodbye to Data Type Conversion in pandas 2.0

```{code-cell} ipython3
!pip install pandas==2.0.0
```

Previously in pandas, if a Series had missing values, its data type would be converted to float, resulting in a potential loss of precision for the original data.

```{code-cell} ipython3
import pandas as pd

s1 = pd.Series([0, 1, 2, 3])
print(f"Data type without None: {s1.dtypes}")

s1.iloc[0] = None
print(f"Data type with None: {s1.dtypes}")
```

With the integration of Apache Arrow in pandas 2.0, this issue is solved.

```{code-cell} ipython3
s2 = pd.Series([0, 1, 2, 3], dtype='int64[pyarrow]')
print(f"Data type without None: {s2.dtypes}")

s2.iloc[0] = None
print(f"Data type with None: {s2.dtypes}")
```

### Efficient String Data Handling in pandas 2.0 with PyArrow Arrays

```{code-cell} ipython3
:tags: [hide-cell]

!pip install 'pandas==2.2' pyarrow
```

As of pandas 2.0, data in pandas can be stored in PyArrow arrays in addition to NumPy arrays. PyArrow arrays provide a wide range of data types compared to NumPy.

One significant advantage of PyArrow arrays is their string datatype, which offers superior speed and memory efficiency than storing strings using object dtypes. 

```{code-cell} ipython3
import pandas as pd
import numpy as np

data_size = 1_000_000
np.random.seed(42)
data = np.random.choice(["John", "Alice", "Michael"], size=data_size)
s_numpy = pd.Series(data)
s_pyarrow = pd.Series(data, dtype="string[pyarrow]")
```

```{code-cell} ipython3
print(f"Datatype of Series with Numpy backend: {s_numpy.dtype}")
print(f"Datatype of Series with PyArrow backend: {s_pyarrow.dtype}")
```

```{code-cell} ipython3
numpy_memory = s_numpy.memory_usage(deep=True)
pyarrow_memory = s_pyarrow.memory_usage(deep=True)

print(f"Memory usage for Numpy backend: {numpy_memory / (1024 ** 2):.2f} MB.")
print(f"Memory usage for PyArrow backend: {pyarrow_memory / (1024 ** 2):.2f} MB.")
print(f"PyArrow backend consumes approximately {numpy_memory / pyarrow_memory:.2f} times less memory than Numpy backend.")
```
