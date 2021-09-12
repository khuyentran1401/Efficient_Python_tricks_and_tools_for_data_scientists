#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[39]:


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# ## Change Values

# pandas' methods to change the values of a pandas DataFrame or a pandas Series. 

# ### pandas.DataFrame.agg: Aggregate over Columns or Rows Using Multiple Operations

# If you want to aggregate over columns or rows using one or more operations, try `pd.DataFrame.agg`.

# In[5]:


from collections import Counter
import pandas as pd


def count_two(nums: list):
    return Counter(nums)[2]


df = pd.DataFrame({"coll": [1, 3, 5], "col2": [2, 4, 6]})
df.agg(["sum", count_two])


# ### pandas.DataFrame.agg: Apply Different Aggregations to Different Columns

# If you want to apply different aggregations to different columns, insert a dictionary of column and aggregation methods to the `pd.DataFrame.agg` method.

# In[6]:


df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 3, 4, 5]})

df.agg({"a": ["sum", "mean"], "b": ["min", "max"]})


# ### pandas.DataFrame.pipe: Increase the Readability of your Code when Applying Multiple Functions to a DataFrame

# If you want to increase the readability of your code when applying multiple functions to a DataFrame, use `pands.DataFrame.pipe` method.

# In[8]:


from textblob import TextBlob

def remove_white_space(df: pd.DataFrame):
    df['text'] = df['text'].apply(lambda row: row.strip())
    return df

def get_sentiment(df: pd.DataFrame):
    df['sentiment'] = df['text'].apply(lambda row:
                                    TextBlob(row).sentiment[0])
    return df

df = pd.DataFrame({'text': ["It is a beautiful day today  ",
                        "  This movie is terrible"]})

df = (df.pipe(remove_white_space)
    .pipe(get_sentiment)
)

df


# ### pandas.Series.map: Change Values of a Pandas Series Using a Dictionary	

# If you want to change values of a pandas Series using a dictionary, use `pd.Series.map`.

# In[9]:


s = pd.Series(["a", "b", "c"])

s.map({"a": 1, "b": 2, "c": 3})


# ### pandas.Series.str: Manipulate Text Data in a Pandas Series	

# If you are working the text data in a pandas Series, instead of creating your own functions, use `pandas.Series.str` to access common methods to process string.
# 
# The code below shows how to convert text to lower case then replace “e” with “a”.

# In[10]:


fruits = pd.Series(['Orange', 'Apple', 'Grape'])
fruits


# In[11]:


fruits.str.lower()


# In[14]:


fruits.str.lower().str.replace("e", "a")


# Find other useful string methods [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html#string-methods).

# ### set_categories in Pandas: Sort Categorical Column by a Specific Ordering

# If you want to sort pandas DataFrame’s categorical column by a specific ordering such as small, medium, large, use `df.col.cat.set_categories()` method.

# In[40]:


df = pd.DataFrame(
    {"col1": ["large", "small", "mini", "medium", "mini"], "col2": [1, 2, 3, 4, 5]}
)
ordered_sizes = "large", "medium", "small", "mini"

df.col1 = df.col1.astype("category")
df.col1.cat.set_categories(ordered_sizes, ordered=True, inplace=True)
df.sort_values(by="col1")


# ### parse_dates: Convert Columns into Datetime When Using Pandas to Read CSV Files

# If there are datetime columns in your csv file, use the `parse_dates` parameter when reading csv file with pandas. This reduces one extra step to convert these columns from string to datetime after reading the file.

# In[52]:


df = pd.read_csv("data1.csv", parse_dates=["date_column_1", "date_column_2"])


# In[53]:


df


# In[54]:


df.info()


# ### Pandas.Series.isin: Filter Rows Only If Column Contains Values From Another List

# When working with a pandas Dataframe, if you want to select the rows when a column contains values from another list, the fastest way is to use `isin`. 
# 
# In the example below, row `2` is filtered out because `3` is not in the list.

# In[23]:


df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
df


# In[27]:


l = [1, 2, 6, 7]
df.a.isin(l)


# In[28]:


df = df[df.a.isin(l)]
df


# ### Specify Suffixes When Using df.merge()

# If you are merging 2 dataframes that have the same features using `df.merge()`, it might be confusing to know which dataframe `a_x` or `a_y` belongs to.

# In[31]:


df1 = pd.DataFrame({"left_key": [1, 2, 3], "a": [4, 5, 6]})
df2 = pd.DataFrame({"right_key": [1, 2, 3], "a": [5, 6, 7]})
df1.merge(df2, left_on="left_key", right_on="right_key")


# A better way is to specify suffixes of the features in each Dataframe like below. Now `a_x` becomes `a_left` and `a_y` becomes `a_right`.

# In[30]:


df1.merge(df2, left_on="left_key", right_on="right_key", suffixes=("_left", "_right"))


# Try it if you want the names of your columns to be less confusing.

# ### Highlight your pandas DataFrame

# Have you ever wanted to highlight your pandas DataFrame to analyze it easier? For example, positive values will be highlighted as green and negative values will be highlighted as red.
# 
# That could be done with `df.style.apply(highlight_condition_func)`. 

# In[32]:


df = pd.DataFrame({"col1": [-5, -2, 1, 4], "col2": [2, 3, -1, 4]})


# In[33]:


def highlight_number(row):
    return [
        "background-color: red; color: white"
        if cell <= 0
        else "background-color: green; color: white"
        for cell in row
    ]


# In[34]:


df.style.apply(highlight_number)


# ### Assign Values to Multiple New Columns

# If you want to assign values to multiple new columns, instead of assigning them separately, you can do everything in one line of code with `df.assign`.
# 
# In the code below, I first created `col3` then use `col3` to create `col4`. Everything is in one line of code.

# In[35]:


df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

df = df.assign(col3=lambda x: x.col1 * 100 + x.col2).assign(
    col4=lambda x: x.col2 * x.col3
)
df


# ### Reduce pandas.DataFrame’s Memory

# If you want to reduce the memory of your pandas DataFrame, start with changing the data type of a column. If your categorical variable has low cardinality, change the data type to category like below.

# In[44]:


from sklearn.datasets import load_iris

X, y = load_iris(as_frame=True, return_X_y=True)
df = pd.concat([X, pd.DataFrame(y, columns=["target"])], axis=1)
df.memory_usage()


# In[43]:


df["target"] = df["target"].astype("category")
df.memory_usage()


# The memory is now is reduced to almost a fifth of what it was!

# ### pandas.DataFrame.explode: Transform Each Element in an Iterable to a Row

# When working with `pandas DataFrame`, if you want to transform each element in an iterable to a row, use `explode`.

# In[45]:


df = pd.DataFrame({"a": [[1, 2], [4, 5]], "b": [11, 13]})
df


# In[46]:


df.explode("a")


# ### pandas.cut: Bin a DataFrame’s values into Discrete Intervals

# If you want to bin your Dataframe’s values into discrete intervals, use `pd.cut`.

# In[47]:


df = pd.DataFrame({"a": [1, 3, 7, 11, 14, 17]})

bins = [0, 5, 10, 15, 20]
df["binned"] = pd.cut(df["a"], bins=bins)

df


# ### Forward Fill in pandas: Use the Previous Value to Fill the Current Missing Value

# If you want to use the previous value in a column or a row to fill the current missing value in a pandas DataFrame, use `df.fillna(method=’ffill’)`. `ffill` stands for forward fill.

# In[48]:


import numpy as np

df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, np.nan], "c": [1, 2, 3]})
df


# In[49]:


df = df.fillna(method="ffill")
df


# ### pandas.pivot_table: Turn Your DataFrame Into a Pivot Table

# A pivot table is useful to summarize and analyze the patterns in your data. If you want to turn your DataFrame into a pivot table, use `pandas.pivot_table`.

# In[50]:


df = pd.DataFrame(
    {
        "item": ["apple", "apple", "apple", "apple", "apple"],
        "size": ["small", "small", "large", "large", "large"],
        "location": ["Walmart", "Aldi", "Walmart", "Aldi", "Aldi"],
        "price": [3, 2, 4, 3, 2.5],
    }
)

df


# In[51]:


pivot = pd.pivot_table(
    df, values="price", index=["item", "size"], columns=["location"], aggfunc="mean"
)
pivot

