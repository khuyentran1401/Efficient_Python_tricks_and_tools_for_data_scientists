#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Tools to Speed Up Code

# This section covers some tools to speed up your code.

# ### Fastai's df_shrink: Shrink DataFrame's Memory Usage in One Line of Code

# Changing data types of DataFrame columns to smaller data types can significantly reduce the memory usage of the DataFrame. Instead of manually choosing smaller data types, is there a way that you can automatically change data types in one line of code?
# 
# That is when the `df_shrink` method of Fastai comes in handy. In the code below, the memory usage of the DataFrame decreases from 200 bytes to 146 bytes.

# In[4]:


from fastai.tabular.core import df_shrink
import pandas as pd

df = pd.DataFrame({"col1": [1, 2, 3], "col2": [1.0, 2.0, 3.0]})
print(df.info())


# In[5]:


new_df = df_shrink(df)
print(new_df.info())


# [Link to Fastai](https://docs.fast.ai/).

# ### Swifter: Add One Word to Make Your Pandas Apply 23 Times Faster

# If you want to have faster pandas apply when working with large data, try swifter. To use swifter, simply add `.swifter` before `.apply`. Everything else is the same.

# In the code below, I compared the speed of Pandas' `apply` and the speed of swifter's `apply` using the California housing dataset of 20640 rows.

# In[7]:


from time import time
from sklearn.datasets import fetch_california_housing
from scipy.special import boxcox1p
import swifter
import timeit

X, y = fetch_california_housing(return_X_y=True, as_frame=True)


def pandas_apply():
    X["AveRooms"].apply(lambda x: boxcox1p(x, 0.25))


def swifter_apply():
    X["AveRooms"].swifter.apply(lambda x: boxcox1p(x, 0.25))


num_experiments = 100
pandas_time = timeit.timeit(pandas_apply, number=num_experiments)
swifter_time = timeit.timeit(swifter_apply, number=num_experiments)

pandas_vs_swifter = round(pandas_time / swifter_time, 2)
print(f"Swifter apply is {pandas_vs_swifter} times faster than Pandas apply")


# Using swifter apply is 23.56 times faster than Pandas apply! This ratio is calculated by taking the average run time of each method after 100 experiments.
