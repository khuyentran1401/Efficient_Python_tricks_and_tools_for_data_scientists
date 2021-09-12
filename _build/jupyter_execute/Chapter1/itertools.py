#!/usr/bin/env python
# coding: utf-8

# ## Itertools

# [itertools](https://docs.python.org/3/library/itertools.html) is a built-in Python library that creates iterators for efficient looping. This section will show you some useful methods of itertools.

# ### itertools.combinations: A Better Way to Iterate Through a Pair of Values in a Python List

# If you want to iterate through a pair of values in a list and the order does not matter (`(a,b)` is the same as `(b, a)`), a naive approach is to use two for-loops.

# In[1]:


num_list = [1, 2, 3]


# In[2]:


for i in num_list: 
    for j in num_list:
        if i < j:
            print((i, j))


# However, using two for-loops is lengthy and inefficient. Use `itertools.combinations` instead:

# In[3]:


from itertools import combinations

comb = combinations(num_list, 2) # use this
for pair in list(comb):
    print(pair)


# ### itertools.product: Nested For-Loops in a Generator Expression 

# Are you using nested for-loops to experiment with different combinations of parameters? If so, use `itertools.product` instead.
# 
# `itertools.product` is more efficient than nested loop because `product(A, B)` returns the same as `((x,y) for x in A for y in B)`.

# In[4]:


from itertools import product

params = {
    "learning_rate": [1e-1, 1e-2, 1e-3],
    "batch_size": [16, 32, 64],
}

for vals in product(*params.values()):
    combination = dict(zip(params.keys(), vals))
    print(combination)

