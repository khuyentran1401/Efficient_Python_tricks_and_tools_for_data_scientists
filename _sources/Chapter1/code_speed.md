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

## Code Speed

+++

This section will show you some ways to speed up or track the performance of your Python code. 

+++

### Concurrently Execute Tasks on Separate CPUs	

+++

If you want to concurrently execute tasks on separate CPUs to run faster, consider using `joblib.Parallel`. It allows you to easily execute several tasks at once, with each task using its own processor.

```{code-cell} ipython3
from joblib import Parallel, delayed
import multiprocessing

def add_three(num: int):
    return num + 3

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(add_three)(i) for i in range(10))
results 
```

### Compare The Execution Time Between 2 Functions

+++

If you want to compare the execution time between 2 functions, try `timeit.timeit`. You can also specify the number of times you want to rerun your function to get a better estimation of the time.

```{code-cell} ipython3
import time 
import timeit 

def func():
    """comprehension"""
    l = [i for i in range(10_000)]

def func2():
    """list range"""
    l = list(range(10_000))

expSize = 1000
time1 = timeit.timeit(func, number=expSize)
time2 = timeit.timeit(func2, number=expSize)

print(time1/time2)
```

From the result, we can see that it is faster to use list range than to use list comprehension on average.

+++

### Save Disk Space on Large Datasets with Parquet 

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pyarrow
```

To save disk space on large datasets, use Parquet files instead of CSV.  Because Parquet files are compressed, they take up less space on disk and in memory than uncompressed CSV files. 

For a 1 million row, 10 column dataset, storing it as CSV takes about 189.59 MB, while storing it as Parquet takes around 78.96 MB, saving approximately 110.63 MB of storage.

```{code-cell} ipython3
import numpy as np
import pandas as pd

# Create a dataset with 1 million rows and 10 columns 
np.random.seed(123)
data = np.random.randint(0, 2**63, size=(1000000, 10))
df = pd.DataFrame(data, columns=[f'col{str(i)}' for i in range(10)])
```

```{code-cell} ipython3
# Write data to Parquet file
df.to_parquet('example.parquet')

# Write data to CSV file
df.to_csv('example.csv', index=False)
```

```{code-cell} ipython3
import os
print("Parquet file size:", os.path.getsize('example.parquet'))
print("CSV file size:", os.path.getsize('example.csv')) 
```
