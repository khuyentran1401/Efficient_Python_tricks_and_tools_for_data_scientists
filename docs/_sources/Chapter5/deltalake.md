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

## Delta Lake

+++

### Version Your Pandas DataFrame with Delta Lake

```{code-cell} ipython3
:tags: [hide-cell]

!pip install deltalake
```

Versioning your data is essential to undoing mistakes, preventing data loss, and ensuring reproducibility. Delta Lake makes it easy to version pandas DataFrames and review past changes for auditing and debugging purposes.

+++

To version a pandas DataFrame with Delta Lake, start with writing out a pandas DataFrame to a Delta table. 

```{code-cell} ipython3
import pandas as pd
import os
from deltalake.writer import write_deltalake

df = pd.DataFrame({"x": [1, 2, 3]})

# Write to a delta table 
table = "delta_lake"
os.makedirs(table, exist_ok=True)
write_deltalake(table, df)
```

Delta Lake stores the data in a Parquet file and maintains a transaction log that records the data operations, enabling time travel and versioning.

+++

```bash
delta_lake:

 ├──  0-4719861e-1d3a-49f8-8870-225e4e46e3a0-0.parquet  
 └──  _delta_log/ 
 │  └────  00000000000000000000.json  
```

+++

To load the Delta table as a pandas DataFrame, simply use the `DeltaTable` object:

```{code-cell} ipython3
from deltalake import DeltaTable

dt = DeltaTable(table)
dt.to_pandas()
```

Let's see what happens when we append another pandas DataFrame to the Delta table.

```{code-cell} ipython3
df2 = pd.DataFrame({"x": [8, 9, 10]})

write_deltalake(table, df2, mode="append")
```

```{code-cell} ipython3
# Create delta table
dt = DeltaTable(table)
dt.to_pandas()
```

Our Delta table now has two versions. Version 0 contains the initial data and Version 1 includes the data that was appended.

![](../img/delta_lake.png)

+++

To get the metadata of files that currently make up the current table such as creation time, size, and statistics, call the `get_add_actions` method.  

```{code-cell} ipython3
dt.get_add_actions(flatten=True).to_pandas()
```

To access prior versions, simply specify the version number when loading the Delta table:

```{code-cell} ipython3
# Read Version 0 of the dataset
dt0 = DeltaTable(table, version=0)
dt0.to_pandas()
```

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Beyond Parquet: Reliable Data Storage with Delta Lake

+++

Traditional data storage methods, such as plain Parquet files, are susceptible to partial failures during write operations. This can result in incomplete data files and a lack of clear recovery options in the event of a system crash.

Delta Lake's write operation with ACID transactions helps solve this by:
- Ensuring either all data is written successfully or none of it is
- Maintaining a transaction log that tracks all changes
- Providing time travel capabilities to recover from failures

+++

Here's an example showing Delta Lake's reliable write operation:

```{code-cell} ipython3
from deltalake import write_deltalake, DeltaTable
import pandas as pd

initial_data = pd.DataFrame({
    "id": [1, 2],
    "value": ["a", "b"]
})

write_deltalake("customers", initial_data)
```

If the append operation fails halfway, Delta Lake's transaction log ensures that the table remains in its last valid state. 

```{code-cell} ipython3
try:
    # Simulate a large append that fails halfway
    new_data = pd.DataFrame({
        "id": range(3, 1003),  # 1000 new rows
        "value": ["error"] * 1000
    })
    
    # Simulate system crash during append
    raise Exception("System crash during append!")
    write_deltalake("customers", new_data, mode="append")
    
except Exception as e:
    print(f"Write failed: {e}")
    
    # Check table state - still contains only initial data
    dt = DeltaTable("customers")
    print("\nTable state after failed append:")
    print(dt.to_pandas())
    
    # Verify version history
    print(f"\nCurrent version: {dt.version()}")
```

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Optimize Query Speed with Data Partitioning

+++

Partitioning data allows queries to target specific segments rather than scanning the entire table, which speeds up data retrieval.

The following code uses Delta Lake to select partitions from a pandas DataFrame. Partitioned data loading is approximately 24.5 times faster than loading the complete dataset and then querying a particular subset

```{code-cell} ipython3
import pandas as pd
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
from datetime import datetime
import numpy as np
```

```{code-cell} ipython3
# Create a DataFrame with hourly sales data for 2 million records
np.random.seed(0)  # For reproducibility

start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 8, 31)
date_range = pd.date_range(start_date, end_date, freq='H')

data = {
    'datetime': date_range,
    'value': np.random.randint(100, 1000, len(date_range))
}

df = pd.DataFrame(data)
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
```

```{code-cell} ipython3
df[["month", "day", "hour", "value"]].head(5)
```

```{code-cell} ipython3
# Write to a Delta table
table_path = 'delta_lake'
write_deltalake(table_path, df)
```

```{code-cell} ipython3
%%timeit
# Load the data from the Delta table
DeltaTable(table_path).to_pandas().query("month == 1 & day == 1")
```

```{code-cell} ipython3
# Write to a Delta table
table_path = "delta_lake2"
write_deltalake(table_path, df, partition_by=["month", "day"])
```

```{code-cell} ipython3
%%timeit
# Load the data from the Delta table
DeltaTable(table_path).to_pandas([("month", "=", "1"), ("day", "=", "1")])
```

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Overwrite Partitions of a pandas DataFrame

```{code-cell} ipython3
:tags: [hide-cell]

!pip install deltalake
```

If you need to modify a specific subset of your pandas DataFrame, such as yesterday's data, it is not possible to overwrite only that partition. Instead, you have to load the entire DataFrame into memory as a workaround solution.

Delta Lake makes it easy to overwrite partitions of a pandas DataFrame.

+++

First, write out a pandas DataFrame as a Delta table that is partitioned by the `date` column.

```{code-cell} ipython3
import pandas as pd
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
```

```{code-cell} ipython3
table_path = "tmp/records" 
df = pd.DataFrame(
    {"a": [1, 2, 3], "date": ["04-21", "04-22", "04-22"]}
)
write_deltalake(
    table_path,
    df,
    partition_by=["date"],
)
```

The Delta table's contents are partitioned by date, with each partition represented by a directory
 
```bash
 └──  _delta_log/ 
 │  └────  00000000000000000000.json  
 └──  date=04-21/ 
 │  └────  0-a6813d0c-157b-4ca6-8b3c-8d5afd51947c-0.parquet  
 └──  date=04-22/ 
 │  └────  0-a6813d0c-157b-4ca6-8b3c-8d5afd51947c-0.parquet  
```

+++

View the Delta table as a pandas DataFrame:

```{code-cell} ipython3
DeltaTable(table_path).to_pandas()
```

Next, create another DataFrame with two other records on 04-22. Overwrite the 04-22 partition with the new DataFrame and leave other partitions untouched.

```{code-cell} ipython3
df = pd.DataFrame(
    {"a": [7, 8], "date": ["04-22", "04-22"]}
)
write_deltalake(
    table_path,
    df,
    mode="overwrite",
    partition_filters=[("date", "=", "04-22")],
)
```

```{code-cell} ipython3
DeltaTable(table_path).to_pandas()
```

Here is the updated contents of the Delta table:

```bash
 └──  _delta_log/ 
 │  └────  00000000000000000000.json
 │  └────  00000000000000000001.json    
 └──  date=04-21/ 
 │  └────  0-a6813d0c-157b-4ca6-8b3c-8d5afd51947c-0.parquet  
 └──  date=04-22/ 
 │  ├────  0-a6813d0c-157b-4ca6-8b3c-8d5afd51947c-0.parquet  
 │  └────  1-b5c9640f-f386-4754-b28f-90e361ab4320-0.parquet 
```

+++

Since the data files are not physically removed from disk, you can time travel to the initial version of the data.

```{code-cell} ipython3
DeltaTable(table_path, version=0).to_pandas()
```

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Efficient Data Appending in Parquet Files: Delta Lake vs. Pandas

```{code-cell} ipython3
:tags: [hide-cell]

!pip install deltalake
```

Appending data to an existing Parquet file using pandas involves:
- Loading the entire existing table into memory.
- Merging the new data with the existing table.
- Writing the merged data to the existing file.

This process can be time-consuming and memory-intensive.

```{code-cell} ipython3
import pandas as pd  

df1 = pd.DataFrame([
    (1, "John", 5000),
    (2, "Jane", 6000),
], columns=["employee_id", "employee_name", "salary"])

df2 = pd.DataFrame([
    (3, "Alex", 8000),
], columns=["employee_id", "employee_name", "salary"])
```

```{code-cell} ipython3
# Save to a parquet file
df1.to_parquet("data.parquet")

# Read the data
existing_data = pd.read_parquet("data.parquet")

# Concat two dataframes
df3 = pd.concat([df1, df2])

# Save to a file
df3.to_parquet("data.parquet")
```

Delta Lake offers a more efficient approach to handling this process. With Delta Lake, you can add, remove, or modify columns without the need to recreate the entire table.

Delta Lake is also built on top of the Parquet file format so it retains the efficiency and columnar storage benefits of Parquet. 

```{code-cell} ipython3
from deltalake.writer import write_deltalake

table_path = "employees"

# Write to Delta Lake
write_deltalake(table_path, df1)

# Append to Delta Lake
write_deltalake(table_path, df2, mode="append")
```

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Enforce Data Quality with Delta Lake Constraints

+++

Delta Lake provides a convenient way to enforce data quality by adding constraints to a table, ensuring that only valid and consistent data can be added.

In the provided code, attempting to add new data with a negative salary violates the constraint of a positive salary, and thus, the data is not added to the table.

```{code-cell} ipython3
:tags: [hide-cell]

!pip install deltalake
```

```{code-cell} ipython3
import pandas as pd
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
```

```{code-cell} ipython3
table_path = "delta_lake"
```

```{code-cell} ipython3
df1 = pd.DataFrame(
    [
        (1, "John", 5000),
        (2, "Jane", 6000),
    ],
    columns=["employee_id", "employee_name", "salary"],
)

write_deltalake(table_path, df1)
```

```{code-cell} ipython3
df1
```

```{code-cell} ipython3
table = DeltaTable(table_path)
table.alter.add_constraint({"salary_gt_0": "salary > 0"})
```

```{code-cell} ipython3
df2 = pd.DataFrame(
    [(3, "Alex", -200)],
    columns=["employee_id", "employee_name", "salary"],
)

write_deltalake(table, df2, mode="append", engine="rust")
```

```bash
DeltaProtocolError: Invariant violations: ["Check or Invariant (salary > 0) violated by value in row: [3, Alex, -200]"]
```

+++

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Efficient Data Updates and Scanning with Delta Lake

```{code-cell} ipython3
:tags: [hide-cell]

!pip install -U "deltalake==0.10.1"
```

Every time new data is appended to an existing Delta table, a new Parquet file is generated. This allows data to be ingested incrementally without having to rewrite the entire dataset.

As files accumulate, read operations may surge. The compact function merges small files into larger ones, enhancing scanning performance.

Combining incremental processing with the compact function enables efficient data updates and scans as your dataset expands.

```{code-cell} ipython3
import pandas as pd
from deltalake.writer import write_deltalake

table_path = 'delta_lake'
data_url = "https://gist.githubusercontent.com/khuyentran1401/458905fc5c630d7a1f7a510a04e5e0f9/raw/5b2d760011c9255a68eb08b83b3b8759ffa25d5c/data.csv"
dfs = pd.read_csv(data_url, chunksize=100)
for df in dfs:
    write_deltalake(table_path, df, mode="append")
```

```{code-cell} ipython3
from deltalake import DeltaTable

dt = DeltaTable(table_path)
```

```{code-cell} ipython3
%%timeit
df = dt.to_pandas()
```

```{code-cell} ipython3
dt.optimize.compact()
```

```{code-cell} ipython3
%%timeit
df = dt.to_pandas()
```

[Link to Delta Lake](https://github.com/delta-io/delta-rs).

+++

### Simplify Table Merge Operations with Delta Lake

```{code-cell} ipython3
:tags: [hide-cell]

!pip install delta-spark
```

Merging two datasets and performing both insert and update operations can be a complex task.

Delta Lake makes it easy to perform multiple data manipulation operations during a merge operation.

The following code demonstrates merging two datasets using Delta Lake:
- If a match is found, the `last_talk` column in `people_table` is updated with the corresponding value from `new_df`. 
- If the `last_talk` value in `people_table` is older than 30 days and the corresponding row is not present in the `new_df` table, the `status` column is updated to 'rejected'.

```{code-cell} ipython3
:tags: [hide-cell]

import pyspark
from delta import *

# Configure Spark to use Delta
builder = (
    pyspark.sql.SparkSession.builder.appName("MyApp")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
```

```{code-cell} ipython3
:tags: [hide-cell]

# Create a spark dataframe
data = [
    (0, "A", "2023-04-15", "interviewing"),
    (1, "B", "2023-05-01", "interviewing"),
    (2, "C", "2023-03-01", "interviewing"),

]

df = (
    spark.createDataFrame(data)
    .toDF("id", "company", "last_talk", "status")
    .repartition(1)
)

# Write to a delta table
path = "tmp/interviews"
df.write.format("delta").save(path)
```

```{code-cell} ipython3
:tags: [hide-cell]

from delta.tables import DeltaTable

# Update the delta table
people_table = DeltaTable.forPath(spark, path)
```

```{code-cell} ipython3
# Target table
people_table.toDF().show()
```

```{code-cell} ipython3
:tags: [hide-cell]

new_data = [(0, "A", "2023-05-07")]
new_df = (
    spark.createDataFrame(new_data).toDF("id", "company", "last_talk").repartition(1)
)
```

```{code-cell} ipython3
# Source table
new_df.show()
```

```{code-cell} ipython3
:tags: [remove-output]

one_month_ago = "current_date() - INTERVAL '30' DAY"

people_table.alias("target").merge(
    new_df.alias("source"), "target.id = source.id"
).whenMatchedUpdate(
    set={"target.last_talk": "source.last_talk", "target.status": "'interviewing'"}
).whenNotMatchedBySourceUpdate(
    condition=f"target.last_talk <= {one_month_ago}",
    set={"target.status": "'rejected'"},
).execute()
```

```{code-cell} ipython3
people_table.toDF().show()
```

[Link to Delta Lake](https://github.com/delta-io/delta).

+++

### From Complex SQL to Simple Merges: Delta Lake's Upsert Solution

```{code-cell} ipython3
:tags: [hide-cell]

!pip install delta-spark
```

```{code-cell} ipython3
:tags: [remove-cell]

import pyspark
from delta import *

# Configure Spark to use Delta
builder = (
    pyspark.sql.SparkSession.builder.appName("MyApp")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
```

Traditionally, implementing upsert (update or insert) logic requires separate UPDATE and INSERT statements or complex SQL. This approach can be error-prone and inefficient, especially for large datasets. 

Delta Lake's merge operation solves this problem by allowing you to specify different actions for matching and non-matching records in a single, declarative statement.

Here's an example that demonstrates the power and simplicity of Delta Lake's merge operation:

First, let's set up our initial data:



```{code-cell} ipython3
# Create sample data for 'customers' DataFrame
customers_data = [
    (1, "John Doe", "john@example.com", "2023-01-01 10:00:00"),
    (2, "Jane Smith", "jane@example.com", "2023-01-02 11:00:00"),
    (3, "Bob Johnson", "bob@example.com", "2023-01-03 12:00:00"),
]
customers = spark.createDataFrame(
    customers_data, ["customer_id", "name", "email", "last_updated"]
)

# Create sample data for 'updates' DataFrame
updates_data = [
    (2, "Jane Doe", "jane.doe@example.com"),  # Existing customer with updates
    (3, "Bob Johnson", "bob@example.com"),  # Existing customer without changes
    (4, "Alice Brown", "alice@example.com"),  # New customer
]
updates = spark.createDataFrame(updates_data, ["customer_id", "name", "email"])

# Show the initial data
print("Initial Customers:")
customers.show()
print("Updates:")
updates.show()
```

Next, we create a Delta table from our initial customer data:

```{code-cell} ipython3
# Define the path where you want to save the Delta table
delta_table_path = "customers_delta"

# Write the DataFrame as a Delta table
customers.write.format("delta").mode("overwrite").save(delta_table_path)

# Create a DeltaTable object
customers_delta = DeltaTable.forPath(spark, delta_table_path)

print("Customers Delta Table created successfully")
```

Now, here's the key part - the merge operation that handles both updates and inserts in a single statement:

```{code-cell} ipython3
# Assume 'customers_delta' is your target table and 'updates' is your source of new data
customers_delta.alias("target").merge(
    updates.alias("source"),
    "target.customer_id = source.customer_id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "email": "source.email",
    "last_updated": "current_timestamp()"
}).whenNotMatchedInsert(values={
    "customer_id": "source.customer_id",
    "name": "source.name",
    "email": "source.email",
    "last_updated": "current_timestamp()"
}).execute()
```

```{code-cell} ipython3
# Verify the updated data
print("Updated Customers Delta Table:")
customers_delta.toDF().show()
```

### The Best Way to Append Mismatched Data to Parquet Tables

+++

Appending mismatched data to a Parquet table involves reading the existing data, concatenating it with the new data, and overwriting the existing Parquet file. This approach can be expensive and may lead to schema inconsistencies.

In the following code, the datatype of `col3` is supposed to be `int64` instead of `float64`.

```{code-cell} ipython3
:tags: [remove-cell]

import warnings
warnings.simplefilter("ignore", UserWarning)
```

```{code-cell} ipython3
import pandas as pd  

filepath = 'test.parquet'

# Write a dataframe to a parquet file
df1 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df1.to_parquet(filepath)

# Append a dataframe to a parquet file
df2 = pd.DataFrame({'col1': [2], 'col2': [7], 'col3': [0]})
concatenation = pd.concat([df1, df2]) # concatenate dataframes
concatenation.to_parquet(filepath) # overwrite original file
```

```{code-cell} ipython3
concat_df = pd.read_parquet(filepath)
print(concat_df, "\n")
print(concat_df.dtypes)
```

With Delta Lake, you can effortlessly append DataFrames with extra columns while ensuring the preservation of your data's schema.

```{code-cell} ipython3
:tags: [hide-cell]

import pyspark
from delta import *

# Configure Spark to use Delta
builder = (
    pyspark.sql.SparkSession.builder.appName("MyApp")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
```

```{code-cell} ipython3
# Create a spark Dataframe
data = [(1, 3), (2, 4)]

df1 = (
    spark.createDataFrame(data)
    .toDF("col1", "col2")
    .repartition(1)
)

# Write to a delta table
path = "tmp"
df1.write.format("delta").save(path)
```

```{code-cell} ipython3
# Create a new DataFrame
new_data = [(2, 7, 0)]
df2 = (
    spark.createDataFrame(new_data).toDF("col1", "col2", "col3").repartition(1)
)
df2.show()
```

```{code-cell} ipython3
# Append to the existing Delta table
df2.write.option("mergeSchema", "true").mode("append").format("delta").save(path)
```

```{code-cell} ipython3
# Read the Delta table
from delta.tables import DeltaTable

table = DeltaTable.forPath(spark, path)
concat_df = table.toDF().pandas_api()

print(concat_df, "\n")
print(concat_df.dtypes)
```

[Link to Delta Lake](https://github.com/delta-io/delta).
