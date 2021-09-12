#!/usr/bin/env python
# coding: utf-8

# ## Join Iterables

# ### join method: Turn an Iterable into a Python String

# If you want to turn an iterable into a string, use `join()`.
# 
# In the code below, I join elements in the list fruits using “, “.

# In[3]:


fruits = ['apples', 'oranges', 'grapes']

fruits_str = ', '.join(fruits)

print(f"Today, I need to get some {fruits_str} in the grocery store")


# ### Zip: Associate Elements from Two Iterators based on the Order

# If you want to associate elements from two iterators based on the order, combine `list` and `zip`. 

# In[1]:


nums = [1, 2, 3, 4]
string = "abcd"
combinations = list(zip(nums, string))
combinations


# ### Zip Function: Create Pairs of Elements from Two Lists in Python

# If you want to create pairs of elements from two lists, use `zip`. `zip()` function takes iterables and aggregates them in a tuple.
# 
# You can also unzip the list of tuples by using `zip(*list_of_tuples)`.
# 

# In[5]:


nums = [1, 2, 3, 4]
chars = ['a', 'b', 'c', 'd']

comb = list(zip(nums, chars))
comb 


# In[6]:


nums_2, chars_2 = zip(*comb)
nums_2, chars_2

