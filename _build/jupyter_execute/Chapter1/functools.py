#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Functools

# [functools](https://docs.python.org/3/library/functools.html) is a built-in Python library to work with functions efficiently. This section will show you some useful methods of functools. 

# ### functools.partial: Generate a New Function with Fewer Arguments

# If you want to fix some arguments of a function and generate a new function with fewer arguments, use `functools.partial`.
# 
# In the code below, I use `partial` to create a new function with only `x` as the argument.

# In[2]:


from functools import partial


def linear_func(x, a, b):
    return a * x + b


linear_func_partial = partial(linear_func, a=2, b=3)
print(linear_func_partial(2))
print(linear_func_partial(4))


# ### functools.singledispatch: Call Another Function Based on the Type of the Current Function’s Argument

# Normally, to call another function based on the type of the current function’s argument, we use an if-else statement:

# In[3]:


data = {"a": [1, 2, 3], "b": [4, 5, 6]}
data2 = [{"a": [1, 2, 3]}, {"b": [4, 5, 6]}]


# In[4]:


def process_data(data):
    if isinstance(data, dict):
        process_dict(data)

    else:
        process_list(data)


def process_dict(data: dict):
    print("Dict is processed")


def process_list(data: list):
    print("List is processed")


# In[5]:


process_data(data)


# In[6]:


process_data(data2)


# With `singledispatch`, you don’t need to use an if-else statement to call an appropriate function. `singledispatch` will choose the right function based on the type of current function’s first argument.

# In[7]:


from functools import singledispatch

@singledispatch
def process_data2(data):
    raise NotImplementedError("Please implement process_data2")


@process_data2.register
def process_dict2(data: dict):
    print("Dict is processed")


@process_data2.register
def process_list2(data: list):
    print("List is processed")


# In[8]:


process_data2(data)


# In[9]:


process_data2(data2)

