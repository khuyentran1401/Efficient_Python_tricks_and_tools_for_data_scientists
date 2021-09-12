#!/usr/bin/env python
# coding: utf-8

# ## Unpack Iterables

# ### How to Unpack Iterables in Python	

# To assign items of a Python iterables (such as list, tuple, string) to different variables, you can unpack the iterable like below.

# In[1]:


nested_arr = [[1, 2, 3], ["a", "b"], 4]
num_arr, char_arr, num = nested_arr


# In[2]:


num_arr


# In[3]:


char_arr


# ### Extended Iterable Unpacking: Ignore Multiple Values when Unpacking a Python Iterable

# If you want to ignore multiple values when unpacking a Python iterable, add `*` to `_` as shown below.
# 
# This is called “Extended Iterable Unpacking” and is available in Python 3.x.

# In[4]:


a, *_, b = [1, 2, 3, 4]
print(a)


# In[5]:


b


# In[6]:


_

