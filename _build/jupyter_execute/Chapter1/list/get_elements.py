#!/usr/bin/env python
# coding: utf-8

# ## Get Elements

# ### random.choice: Get a Randomly Selected Element from a Python List

# Besides getting a random number, you can also get a random element from a Python list using random. In the code below, “stay at home” was picked randomly from a list of options.

# In[1]:


import random 

to_do_tonight = ['stay at home', 'attend party', 'do exercise']

random.choice(to_do_tonight)


# ### random.sample: Get Multiple Random Elements from a Python List

# If you want to get n random elements from a list, use `random.sample`.

# In[2]:


import random

random.seed(1)
nums = [1, 2, 3, 4, 5]
random_nums = random.sample(nums, 2)
random_nums


# ### heapq: Find n Max Values of a Python List

# If you want to extract n max values from a large Python list, using `heapq` will speed up the code.
# 
# In the code below, using heapq is more than 2 times faster than using sorting and indexing. Both methods try to find the max values of a list of 10000 items.
# 

# In[3]:


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


# In[4]:


expSize = 1000
n = 100
time_sorting = timeit("get_n_max_sorting(l, n)", number=expSize,
                        globals=globals())
time_heapq = timeit('get_n_max_heapq(l, n)', number=expSize,
                    globals=globals())

ratio = round(time_sorting/time_heapq, 3)
print(f'Run {expSize} experiments. Using heapq is {ratio} times'
' faster than using sorting')

