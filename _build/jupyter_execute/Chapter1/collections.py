#!/usr/bin/env python
# coding: utf-8

# ## Collections

# [collections](https://docs.python.org/3/library/collections.html) is a built-in Python library to deal with Python dictionary efficiently. This section will show you some useful methods of this module. 

# ### collections.Counter: Count The Occurrences of Items in a List

# Counting the occurrences of each item in a list using a for-loop is slow and inefficient. 

# In[1]:


char_list = ['a', 'b', 'c', 'a', 'd', 'b', 'b']


# In[2]:


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

# In[3]:


from collections import Counter

Counter(char_list)


# In my experiment, using `Counter` is more than 2 times faster than using a custom counter.

# In[4]:


from timeit import timeit
import random 

random.seed(0)
num_list = [random.randint(0,22) for _ in range(1000)]

numExp = 100
custom_time = timeit("custom_counter(num_list)", globals=globals())
counter_time = timeit("Counter(num_list)", globals=globals())
print(custom_time/counter_time)


# ### namedtuple: A Lightweight Python Structure to Mange your Data

# If you need a small class to manage data in your project, consider using namedtuple.
# 
# `namedtuple` object is like a tuple but can be used as a normal Python class.
# 
# In the code below, I use `namedtuple` to create a `Person` object with attributes `name` and `gender`.

# In[1]:


from collections import namedtuple

Person = namedtuple("Person", "name gender")

oliver = Person("Oliver", "male")
khuyen = Person("Khuyen", "female")


# Just like Python class,  you can access attributes of `namedtuple` using `obj.attr`.

# ### Defaultdict: Return a Default Value When a Key is Not Available

# If you want to create a Python dictionary with default value, use `defaultdict`. When calling a key that is not in the dictionary, the default value is returned.

# In[2]:


from collections import defaultdict

classes = defaultdict(lambda: 'Outside')
classes['Math'] = 'B23'
classes['Physics'] = 'D24'
classes['Math']


# In[3]:


classes['English']


# ### Defaultdict: Create a Dictionary with Values that are List

# If you want to create a dictionary with the values that are list, the cleanest way is to pass a list class to a `defaultdict`.

# In[4]:


from collections import defaultdict

# Instead of this
food_price = {'apple': [], 'orange': []}

# Use this
food_price = defaultdict(list)

for i in range(1, 4):
    food_price['apple'].append(i)
    food_price['orange'].append(i)    

print(food_price.items()) 

