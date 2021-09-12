#!/usr/bin/env python
# coding: utf-8

# ## Testing

# This section shows how to compare between 2 Pandas DataFrame or between 2 Pandas Series

# ### assert_frame equal: Test Whether Two DataFrames are Similar

# If you want to test whether two DataFrames are similar or how much they are different from each other, try `pandas.testing.assert_frame_equal`.

# In[1]:


from pandas.testing import assert_frame_equal
import pandas as pd 


df1 = pd.DataFrame({'coll': [1,2,3], 'col2': [4,5,6]})
df2 = pd.DataFrame({'coll': [1,3,4], 'col2': [4,5,6]})
assert_frame_equal(df1, df2)

