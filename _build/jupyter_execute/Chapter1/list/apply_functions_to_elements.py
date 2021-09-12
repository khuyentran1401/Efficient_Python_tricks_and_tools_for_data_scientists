#!/usr/bin/env python
# coding: utf-8

# ## Apply Functions to Elements in a List

# ### any: Check if Any Element of an Iterable is True

# If you want to check if any element of an iterable is True, use any. In the code below, I use any to find if any element in the text is in uppercase.

# In[1]:


text = "abcdE"
any(c for c in text if c.isupper())


# ### filter: Get the Elements of an Iterable that a Function Returns True

# If you want to get the elements of an iterable that a function returns true, use filter.
# 
# In the code below, I use the filter method to get items that are fruits.

# In[2]:


def get_fruit(val: str):
    fruits = ['apple', 'orange', 'grape']
    if val in fruits:
        return True 
    else:
        return False 

items = ['chair', 'apple', 'water', 'table', 'orange']
fruits = filter(get_fruit, items)
print(list(fruits))


# ### map method: Apply a Function to Each Item of an Iterable

# If you want to apply the given function to each item of a given iterable, use `map`.

# In[7]:


nums = [1, 2, 3]
list(map(str, nums))


# In[8]:


def multiply_by_two(num: float):
    return num * 2


list(map(multiply_by_two, nums))

