#!/usr/bin/env python
# coding: utf-8

# ## Code Speed

# This section will show you some ways to speed up or track the performance of your Python code. 

# ### Concurrently Execute Tasks on Separate CPUs	

# If you want to concurrently execute tasks on separate CPUs to run faster, consider using `joblib.Parallel`. It allows you to easily execute several tasks at once, with each task using its own processor.

# In[3]:


from joblib import Parallel, delayed
import multiprocessing

def add_three(num: int):
    return num + 3

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(add_three)(i) for i in range(10))
results 


# ### Compare The Execution Time Between 2 Functions

# If you want to compare the execution time between 2 functions, try `timeit.timeit`. You can also specify the number of times you want to rerun your function to get a better estimation of the time.

# In[4]:


import time 
import timeit 

def func():
    """comprehension"""
    l = [i for i in range(10_000)]

def func2():
    """list range"""
    l = list(range(10_000))

expSize = 1000
time1 = timeit.timeit(func, number=expSize)
time2 = timeit.timeit(func2, number=expSize)

print(time1/time2)


# From the result, we can see that it is faster to use list range than to use list comprehension on average.
