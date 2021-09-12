#!/usr/bin/env python
# coding: utf-8

# ## Dictionary

# ### Key Parameter in Max(): Find the Key with the Largest Value

# Apply max on a Python dictionary will give you the largest key, not the key with the largest value. If you want to find the key with the largest value, specify that using the `key` parameter in the `max` method.

# In[1]:


birth_year = {"Ben": 1997, "Alex": 2000, "Oliver": 1995}

max(birth_year)


# In[2]:


max_val = max(birth_year, key=lambda k: birth_year[k])
max_val

