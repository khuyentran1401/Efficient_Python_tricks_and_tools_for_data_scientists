#!/usr/bin/env python
# coding: utf-8

# ## Collections

# ### collections.Counter: Count the Occurrences of Items in a List

# Counting the occurrences of each item in a list using a for-loop is slow and inefficient. 

# In[2]:


char_list = ['a', 'b', 'c', 'a', 'd', 'b', 'b']


# In[9]:


def custom_counter(list_: list):
    char_counter = {}
    for char in list_:
        if char not in char_counter:
            char_counter[char] = 1
        else: 
            char_counter[char] += 1

    return char_counter
custom_counter(char_list)


# Using `collections.Counter` is more efficient, and all it takes is one line of code!

# In[10]:


from collections import Counter

Counter(char_list)


# In my experiment, using Counter is more than 2 times faster than using custom counter.

# In[11]:


from timeit import timeit
import random 

random.seed(0)
num_list = [random.randint(0,22) for _ in range(1000)]

numExp = 100
custom_time = timeit("custom_counter(num_list)", globals=globals())
counter_time = timeit("Counter(num_list)", globals=globals())
print(custom_time/counter_time)


# In[ ]:




