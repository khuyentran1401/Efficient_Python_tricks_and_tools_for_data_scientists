#!/usr/bin/env python
# coding: utf-8

# ## List

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


# ### How to Unpack Iterables in Python	

# To assign items of a Python iterables (such as list, tuple, string) to different variables, you can unpack the iterable like below.

# In[3]:


nested_arr = [[1, 2, 3], ["a", "b"], 4]
num_arr, char_arr, num = nested_arr


# In[4]:


num_arr


# In[5]:


char_arr


# ### Extended Iterable Unpacking: Ignore Multiple Values when Unpacking a Python Iterable

# If you want to ignore multiple values when unpacking a Python iterable, add `*` to `_` as shown below.
# 
# This is called “Extended Iterable Unpacking” and is available in Python 3.x.

# In[6]:


a, *_, b = [1, 2, 3, 4]
print(a)


# In[7]:


b


# In[8]:


_


# ### random.choice: Get a Randomly Selected Element from a Python List

# Besides getting a random number, you can also get a random element from a Python list using random. In the code below, “stay at home” was picked randomly from a list of options.

# In[9]:


import random 

to_do_tonight = ['stay at home', 'attend party', 'do exercise']

random.choice(to_do_tonight)


# ### random.sample: Get Multiple Random Elements from a Python List

# In[10]:


# TODO: add example


# ### heapq: Find n Max Values of a Python List

# If you want to extract n max values from a large Python list, using `heapq` will speed up the code.
# 
# In the code below, using heapq is more than 2 times faster than using sorting and indexing. Both methods try to find the max values of a list of 10000 items.
# 

# In[11]:


import heapq
import random
from timeit import timeit

random.seed(0)
l = random.sample(range(0, 10000), 10000)

def get_n_max_sorting(l: list, n: int):
    l = sorted(l, reverse=True)
    return l[:n]

def get_n_max_heapq(l: list, n: int):
    return heapq.nlargest(n, l)


# In[12]:


expSize = 1000
n = 100
time_sorting = timeit("get_n_max_sorting(l, n)", number=expSize,
                        globals=globals())
time_heapq = timeit('get_n_max_heapq(l, n)', number=expSize,
                    globals=globals())

ratio = round(time_sorting/time_heapq, 3)
print(f'Run {expSize} experiments. Using heapq is {ratio} times'
' faster than using sorting')


# ### join method: Turn an Iterable into a Python String
