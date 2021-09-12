#!/usr/bin/env python
# coding: utf-8

# ## Tuple

# ### slice: Make Your Indices More Readable by Naming Your Slice

# Have you ever been confused when looking into code that contains hardcoded slice indices? Even if you understand it now, you might forget why you choose specific indices in the future. 

# In[1]:


data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

some_sum = sum(data[:8]) * sum(data[8:])


# If so, name your `slice`. Python provides a nice built-in function for that purpose called `slice`. By using names, your code is much easier to understand.

# In[2]:


JANUARY = slice(0, 8)
FEBRUARY = slice(8, len(data))
some_sum = sum(data[JANUARY] * sum(data[FEBRUARY]))
print(some_sum) 

