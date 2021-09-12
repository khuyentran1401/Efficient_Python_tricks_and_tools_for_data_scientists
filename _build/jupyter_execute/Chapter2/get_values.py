#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Get Values

# This section contains some methods to get specific values of a pandas DataFrame or a pandas Series.

# ### DataFrame.columns.str.startswith: Find DataFrame’s Columns that Start With a Pattern

# To find pandas DataFrame whose columns start with a pattern, use `df.columns.str.startswith`. 

# In[2]:


import pandas as pd 

df = pd.DataFrame({'pricel': [1, 2, 3],
                    'price2': [2, 3, 4],
                    'year': [2020, 2021, 2021]})

mask = df.columns.str.startswith('price')
df.loc[:, mask]


# ### pandas.Series.dt: Access Datetime Properties of a pandas Series

# The easiest way to access datetime properties of pandas Series values is to use `pandas.Series.dt`.

# In[5]:


df = pd.DataFrame({"date": ["2021/05/13 15:00", "2022-6-20 14:00"], "values": [1, 3]})

df["date"] = pd.to_datetime(df["date"])

df["date"].dt.year


# In[6]:


df["date"].dt.time


# ### pd.Series.between: Select Rows in a Pandas Series Containing Values Between 2 Numbers

# To get the values that are smaller than the upper bound and larger than the lower bound, use the `pandas.Series.between` method.
# 
# In the code below, I obtained the values between 0 and 10 using `between`.
# 

# In[7]:


s = pd.Series([5, 2, 15, 13, 6, 10])

s[s.between(0, 10)]


# ### DataFrame rolling: Find The Average of The Previous n Datapoints Using pandas

# If you want to find the average of the previous n data points (simple moving average) with pandas, use `df.rolling(time_period).mean()`.
# 
# The code below shows how to find the simple moving average of the previous 3 data-points.

# In[8]:


from datetime import date

df = pd.DataFrame(
    {
        "date": [
            date(2021, 1, 20),
            date(2021, 1, 21),
            date(2021, 1, 22),
            date(2021, 1, 23),
            date(2021, 1, 24),
        ],
        "value": [1, 2, 3, 4, 5],
    }
).set_index("date")

df


# In[10]:


df.rolling(3).mean()


# ### select_dtypes: Return a Subset of a DataFrame Including/Excluding Columns Based on Their dtype

# You might want to apply different kinds of processing to categorical and numerical features. Instead of manually choosing categorical features or numerical features, you can automatically get them by using `df.select_dtypes('data_type')`.
# 
# In the example below, you can either include or exclude certain data types using `exclude`.
# 
# 

# In[12]:


df = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3], "col3": [0.1, 0.2, 0.3]})

df.info()


# In[14]:


df.select_dtypes(include=["int64", "float64"])


# ### pandas.Series.pct_change: Find The Percentage Change Between The Current and a Prior Element in a pandas Series

# If you want to find the percentage change between the current and a prior element in a pandas Series, use the `pct_change` method.
# 
# In the example below, 35 is 75% larger than 20, and 10 is 71.4% smaller than 35.

# In[15]:


df = pd.DataFrame({"a": [20, 35, 10], "b": [1, 2, 3]})
df


# In[16]:


df.a.pct_change()


# ### DataFrame.diff and DataFrame.shift: Take the Difference Between Rows Within a Column in pandas

# If you want to get the difference between rows within a column, use `DataFrame.diff()`.

# In[18]:


df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 6]})
diff = df.diff()
diff


# This will leave the first index null. You can shift the rows up to match the first difference with the first index using `DataFrame.shift(-1)`.

# In[19]:


shift = diff.shift(-1)
shift


# In[20]:


processed_df = shift.dropna()
processed_df


# ### pandas.clip: Exclude Outliers

# Outliers are unusual values in your dataset, and they can distort statistical analyses. 

# In[25]:


data = {"col0": [9, -3, 0, -1, 5]}
df = pd.DataFrame(data)
df


# If you want to trim values that the outliers, one of the methods is to use `df.clip`.
# 
# Below is how to use the 0.5-quantile as the lower threshold and .95-quantile as the upper threshold

# In[26]:


lower = df.col0.quantile(0.05)
upper = df.col0.quantile(0.95)

df.clip(lower=lower, upper=upper)


# ### Get Rows within a Year Range

# If you want to get all data starting in a particular year and exclude the previous years, simply use `df.loc['year':]` like below. This works when the index of your `pd.Dataframe` is `DatetimeIndex`.

# In[27]:


from datetime import datetime

df = pd.DataFrame(
    {
        "date": [datetime(2018, 10, 1), datetime(2019, 10, 1), datetime(2020, 10, 1)],
        "val": [1, 2, 3],
    }
).set_index("date")

df


# In[28]:


df.loc["2019":]


# ### pandas.reindex: Replace the Values of the Missing Dates with 0

# Have you ever got a time series with missing dates? This can cause a problem since many time series methods require a fixed frequency index.
# 
# To fix this issue, you can replace the values of the missing dates with 0 using `pd.date_range` and `pd.reindex`.

# In[30]:


s = pd.Series([1, 2, 3], index=["2021-07-20", "2021-07-23", "2021-07-25"])
s.index = pd.to_datetime(s.index)
s


# In[31]:


# Get dates ranging from 2021/7/20 to 2021/7/25
new_index = pd.date_range("2021-07-20", "2021-07-25")

# Conform Series to new index
new_s = s.reindex(new_index, fill_value=0)
new_s


# ### Select DataFrame Rows Before or After a Specific Date

# If you want to get the rows whose dates are before or after a specific date, use the comparison operator and a date string.

# In[32]:


df = pd.DataFrame(
    {"date": pd.date_range(start="2021-7-19", end="2021-7-23"), "value": list(range(5))}
)
df


# In[33]:


filtered_df = df[df.date <= "2021-07-21"]
filtered_df


# ### DataFrame.groupby.sample: Get a Random Sample of Items from Each Category in a Column	

# If you want to get a random sample of items from each category in a column, use `pandas.DataFrame.groupby.sample`.This method is useful when you want to get a subset of a DataFrame while keeping all categories in a column.

# In[35]:


df = pd.DataFrame({"col1": ["a", "a", "b", "c", "c", "d"], "col2": [4, 5, 6, 7, 8, 9]})
df.groupby("col1").sample(n=1)


# To get 2 items from each category, use `n=2`.

# In[37]:


df = pd.DataFrame(
    {
        "col1": ["a", "a", "b", "b", "b", "c", "c", "d", "d"],
        "col2": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
)
df.groupby("col1").sample(n=2)

