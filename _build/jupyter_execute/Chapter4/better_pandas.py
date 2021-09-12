#!/usr/bin/env python
# coding: utf-8

# ## Better Pandas

# This section cover tools to make your experience with Pandas a litte bit better.

# ### tqdm: Add Progress Bar to Your Pandas Apply

# If you want to have a progress bar to get updated about the progress of your pandas apply, try tqdm.

# In[1]:


import pandas as pd 
from tqdm import tqdm 
import time 

df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [2, 3, 4, 5, 6]})

tqdm.pandas()
def func(row):
    time.sleep(1)
    return row + 1

df['a'].progress_apply(func)


# [Link to tqdm](https://github.com/tqdm/tqdm).
