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

## Polars

+++

### Polars: Blazing Fast DataFrame Library

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

If you want data manipulation library that's both fast and memory-efficient, try Polars. Polars provides a high-level API similar to Pandas but with better performance for large datasets.

The code below compares the performance of Polars and pandas. 

```{code-cell} ipython3
import pandas as pd
import polars as pl
import numpy as np
import time

# Create two Pandas DataFrames with 1 million rows each
pandas_df1 = pd.DataFrame({
    'key': np.random.randint(0, 1000, size=1_000_000),
    'value1': np.random.rand(1_000_000)
})

pandas_df2 = pd.DataFrame({
    'key': np.random.randint(0, 1000, size=1_000_000),
    'value2': np.random.rand(1000000)
})

# Create two Polars DataFrames from the Pandas DataFrames
polars_df1 = pl.from_pandas(pandas_df1)
polars_df2 = pl.from_pandas(pandas_df2)

# Merge the two DataFrames on the 'key' column
start_time = time.time()
pandas_merged = pd.merge(pandas_df1, pandas_df2, on='key')
pandas_time = time.time() - start_time

start_time = time.time()
polars_merged = polars_df1.join(polars_df2, on='key')
polars_time = time.time() - start_time

print(f"Pandas time: {pandas_time:.6f} seconds")
print(f"Polars time: {polars_time:.6f} seconds")
```

```{code-cell} ipython3
print(f"Polars is {pandas_time/polars_time:.2f} times faster than Pandas")
```

[Link to polars](https://github.com/pola-rs/polars)

+++

### Polars: Speed Up Data Processing 12x with Lazy Execution

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

Polars is a lightning-fast DataFrame library that utilizes all available cores on your machine. 

Polars has two APIs: an eager API and a lazy API.

The eager execution is similar to Pandas, which executes code immediately. 

In contrast, the lazy execution defers computations until the `collect()` method is called. This approach avoids unnecessary computations, making lazy execution potentially more efficient than eager execution.

The code following code shows filter operations on a DataFrame containing 10 million rows. Running polars with lazy execution is 12 times faster than using pandas. 

```{code-cell} ipython3
:tags: [hide-cell]

import numpy as np

# Create a random seed for reproducibility
np.random.seed(42)

# Number of rows in the dataset
num_rows = 10_000_000

# Sample data for categorical columns
categories = ["a", "b", "c", "d"]

# Generate random data for the dataset
data = {
    "Cat1": np.random.choice(categories, size=num_rows),
    "Cat2": np.random.choice(categories, size=num_rows),
    "Num1": np.random.randint(1, 100, size=num_rows),
    "Num2": np.random.randint(1000, 10000, size=num_rows),
}
```

Create a pandas DataFrame and filter the DataFrame. 

```{code-cell} ipython3
import pandas as pd


df = pd.DataFrame(data)
df.head()
```

```{code-cell} ipython3
%timeit df[(df['Cat1'] == 'a') & (df['Cat2'] == 'b') & (df['Num1'] >= 70)]
```

Create a polars DataFrame and filter the DataFrame. 

```{code-cell} ipython3
import polars as pl

pl_df = pl.DataFrame(data)
```

```{code-cell} ipython3
%timeit pl_df.lazy().filter((pl.col('Cat1') == 'a') & (pl.col('Cat2') == 'b') & (pl.col('Num1') >= 70)).collect()
```

[Link to polars](https://github.com/pola-rs/polars)

+++

### Polars vs. Pandas for CSV Loading and Filtering

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

```{code-cell} ipython3
:tags: [hide-cell]

!wget -O airport-codes.csv "https://datahub.io/core/airport-codes/r/0.csv"
```

The `read_csv` method in Pandas loads all rows of the dataset into the DataFrame before filtering to remove all unwanted rows.

On the other hand, the `scan_csv` method in Polars delays execution and optimizes the operation until the `collect` method is called. This approach accelerates code execution, particularly when handling large datasets.

In the code below, it is 25.5 times faster to use Polars instead of Pandas to read a subset of CSV file containing 57k rows. 

```{code-cell} ipython3
import pandas as pd
import polars as pl 
```

```{code-cell} ipython3
%%timeit
df = pd.read_csv("airport-codes.csv")
df[(df["type"] == "heliport") & (df["continent"] == "EU")]
```

```{code-cell} ipython3
%%timeit
pl.scan_csv("airport-codes.csv").filter(
    (pl.col("type") == "heliport") & (pl.col("continent") == "EU")
).collect()
```

### Pandas vs Polars: Harnessing Parallelism for Faster Data Processing

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

Pandas is a single-threaded library, utilizing only a single CPU core. To achieve parallelism with Pandas, you would need to use additional libraries like Dask.

```{code-cell} ipython3
import pandas as pd
import multiprocessing as mp
import dask.dataframe as dd


df = pd.DataFrame({"A": range(1_000_000), "B": range(1_000_000)})

# Perform the groupby and sum operation in parallel 
ddf = dd.from_pandas(df, npartitions=mp.cpu_count())
result = ddf.groupby("A").sum().compute()
```

Polars, on the other hand, automatically leverages the available CPU cores without any additional configuration.

```{code-cell} ipython3
import polars as pl

df = pl.DataFrame({"A": range(1_000_000), "B": range(1_000_000)})

# Perform the groupby and sum operation in parallel 
result = df.group_by("A").sum()
```

[Link to Polars](https://bit.ly/3v9dmCT).

+++

### Simple and Expressive Data Transformation with Polars

+++

Extract features and select only relevant features for each time series.

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

Compared to pandas, Polars provides a more expressive syntax for creating complex data transformation pipelines. Every expression in Polars produces a new expression, and these expressions can be piped together. 

```{code-cell} ipython3
import pandas as pd

df = pd.DataFrame(
    {"A": [1, 2, 6], "B": ["a", "b", "c"], "C": [True, False, True]}
)
integer_columns = df.select_dtypes("int64")
other_columns = df[["B"]]
pd.concat([integer_columns, other_columns], axis=1)
```

```{code-cell} ipython3
import polars as pl

pl_df = pl.DataFrame(
    {"A": [1, 2, 6], "B": ["a", "b", "c"], "C": [True, False, True]}
)
pl_df.select([pl.col(pl.Int64), "B"])
```

### Harness Polars and Delta Lake for Blazing Fast Performance 

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars deltalake
```

Polars is a Rust-based DataFrame library that is designed for high-performance data manipulation and analysis. Delta Lake is a storage format that offers a range of benefits, including ACID transactions, time travel, schema enforcement, and more. It's designed to work seamlessly with big data processing engines like Apache Spark and can handle large amounts of data with ease.


When you combine Polars and Delta Lake, you get a powerful data processing system. Polars does the heavy lifting of processing your data, while Delta Lake keeps everything organized and up-to-date.

Imagine you have a huge dataset with millions of rows. You want to group the data by category and calculate the sum of a certain column. With Polars and Delta Lake, you can do this quickly and easily.

First, you create a sample dataset:

```{code-cell} ipython3
import pandas as pd
import numpy as np

# Create a sample dataset
num_rows = 10_000_000
data = {
    "Cat1": np.random.choice(['A', 'B', 'C'], size=num_rows),
    'Num1': np.random.randint(low=1, high=100, size=num_rows)
}

df = pd.DataFrame(data)
df.head()
```

Next, you save the dataset to Delta Lake:

```{code-cell} ipython3
from deltalake.writer import write_deltalake

save_path = "tmp/data"

write_deltalake(save_path, df)
```

Then, you can use Polars to read the data from Delta Lake and perform the grouping operation:

```{code-cell} ipython3
import polars as pl 

pl_df = pl.read_delta(save_path)

print(pl_df.group_by("Cat1").sum())
```

Let's say you want to append some new data to the existing dataset:

```{code-cell} ipython3
new_data = pd.DataFrame({"Cat1": ["B", "C"], "Num1": [2, 3]})

write_deltalake(save_path, new_data, mode="append")
```

Now, you can use Polars to read the updated data from Delta Lake:

```{code-cell} ipython3
updated_pl_df = pl.read_delta(save_path)
print(updated_pl_df.tail())
```

But what if you want to go back to the previous version of the data? With Delta Lake, you can easily do that by specifying the version number:

```{code-cell} ipython3
previous_pl_df = pl.read_delta(save_path, version=0)
print(previous_pl_df.tail())
```

[Link to polars](https://github.com/pola-rs/polars)

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Parallel Execution of Multiple Files with Polars

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

```{code-cell} ipython3
:tags: [remove-cell]

import polars as pl
from pathlib import Path

Path("test_data").mkdir(parents=True, exist_ok=True)

df1 = pl.DataFrame(
    {"Cat": ["A", "A", "B", "B", "C", "C"], "Num": [1, 1, 2, 2, 3, 3]}
)

df2 = pl.DataFrame(
    {"Cat": ["C", "B", "A", "A", "C"], "Num": [1, 1, 2, 2, 3]}
)

df3 = pl.DataFrame(
    {"Cat": ["A", "C", "B", "B"], "Num": [1, 1, 3, 2]}
)

# Save the dataframes as CSV files
df1.write_csv("test_data/df1.csv")
df2.write_csv("test_data/df2.csv")
df3.write_csv("test_data/df3.csv")
```

If you have multiple files to process, Polars enables you to construct a query plan for each file beforehand. This allows for the efficient execution of multiple files concurrently, maximizing processing speed.

```{code-cell} ipython3
import glob

import polars as pl

# Construct a query plan for each file
queries = []
for file in glob.glob("test_data/*.csv"):
    q = pl.scan_csv(file).group_by("Cat").agg(pl.sum("Num"))
    queries.append(q)

# Execute files in parallel
dataframes = pl.collect_all(queries)
dataframes
```

[Link to polars](https://github.com/pola-rs/polars)

+++

### Polars' Streaming Mode: A Solution for Large Data Sets

```{code-cell} ipython3
:tags: [hide-cell]

!pip install polars
```

```{code-cell} ipython3
:tags: [remove-cell]

!wget https://raw.githubusercontent.com/pola-rs/polars/main/docs/data/reddit.csv 
```

The default collect method in Polars processes your data as a single batch, which means that all the data must fit into your available memory.

If your data requires more memory than you have available, Polars can process it in batches using streaming mode. To use streaming mode, simply pass the `streaming=True` argument to the `collect` method.

```{code-cell} ipython3
import polars as pl

df = (
    pl.scan_csv("reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
    .collect(streaming=True)
)
```

[Learn more about Streaming API in Polars](https://bit.ly/3wlTZXR).

+++

### Pandas vs Polars: Syntax Comparison for Data Scientists

+++

As a data scientist, you're likely familiar with the popular data analysis libraries Pandas and Polars. Both provide powerful tools for working with tabular data, but how do their syntaxes compare?

To begin, we'll create equivalent dataframes in both Pandas and Polars:

```{code-cell} ipython3
import pandas as pd
import polars as pl

# Create a Pandas DataFrame
data = {
    "Category": ["Electronics", "Clothing", "Electronics", "Clothing", "Electronics"],
    "Quantity": [5, 2, 3, 10, 4],
    "Price": [200, 30, 150, 20, 300],
}
pandas_df = pd.DataFrame(data)
polars_df = pl.DataFrame(data)
```

Key Operations Comparison:

```{code-cell} ipython3
pandas_df[["Category", "Price"]]
```

```{code-cell} ipython3
polars_df.select(["Category", "Price"])
```

```{code-cell} ipython3
# Filtering rows where Quantity > 3
pandas_df[pandas_df["Quantity"] > 3]
```

```{code-cell} ipython3
polars_df.filter(pl.col("Quantity") > 3)
```

```{code-cell} ipython3
pandas_df.groupby("Category").agg(
    {
        "Quantity": "sum", 
        "Price": "mean", 
    }
)
```

```{code-cell} ipython3
polars_df.group_by("Category").agg(
    [
        pl.col("Quantity").sum(),
        pl.col("Price").mean(),
    ]
)
```

### Faster Data Analysis with Polars: A Guide to Lazy Execution

+++

When processing data, the execution approach significantly impacts performance. Pandas, a popular Python data manipulation library, uses eager execution by default, processing data immediately and loading everything into memory. This works well for small to medium-sized datasets but can lead to slow computations and high memory usage with large datasets.

In contrast, Polars, a modern data processing library, offers both eager and lazy execution. In lazy mode, a query optimizer evaluates operations and determines the most efficient execution plan, which may involve reordering operations or dropping redundant calculations.

Let's consider an example where we:

- Group a DataFrame by 'region'
- Calculate two aggregations: sum of 'revenue' and count of 'orders'
- Filter for only 'North' and 'South' regions

With eager execution, Pandas will:

- Execute operations immediately, loading all data into memory
- Keep intermediate results in memory during each step
- Execute operations in the exact order written

```{code-cell} ipython3
import numpy as np

# Generate sample data
N = 10_000_000

data = {
    "region": np.random.choice(["North", "South", "East", "West"], N),
    "revenue": np.random.uniform(100, 10000, N),
    "orders": np.random.randint(1, 100, N),
}
```

```{code-cell} ipython3
import pandas as pd


def analyze_sales_pandas(df):
    # Loads and processes everything in memory
    return (
        df.groupby("region")
        .agg({"revenue": "sum"})
        .loc[["North", "South"]]
    )


pd_df = pd.DataFrame(data)
%timeit analyze_sales_pandas(pd_df)
```

As shown above, the eager execution approach used by Pandas results in a execution time of approximately 367 milliseconds.


With lazy execution, Polars will:

- Create an execution plan first, optimizing the entire chain before processing any data
- Only process data once at .collect(), reducing memory overhead
- Rearrange operations for optimal performance (pushing filters before groupby)

```{code-cell} ipython3
import polars as pl


def analyze_sales_polars(df):
    # Creates execution plan, no data processed yet
    result = (
        df.lazy()
        .group_by("region")
        .agg(pl.col("revenue").sum())
        .filter(pl.col("region").is_in(["North", "South"]))
        .collect()  # Only now data is processed
    )
    return result


pl_df = pl.DataFrame(data)
%timeit analyze_sales_polars(pl_df)
```

In contrast, the lazy execution approach with Polars takes approximately 170 milliseconds to complete, which is about 53.68% faster than the eager execution approach with Pandas.
