#!/usr/bin/env python
# coding: utf-8

# ## Efficient Function

# ### **kwargs: Pass Multiple Arguments to a Function in Python

# Sometimes you might not know the arguments you will pass to a function. If so, use `**kwargs`.
# 
# `**kwargs` allow you to pass multiple arguments to a function using a dictionary. In the example below, passing `**{'a':1, 'b':2}` to the function is similar to passing `a=1`, `b=1` to the function.
# 
# Once `**kwargs` argument is passed, you can treat it like a Python dictionary.

# In[1]:


parameters = {'a': 1, 'b': 2}

def example(c, **kwargs):
    print(kwargs)
    for val in kwargs.values():
        print(c + val)

example(c=3, **parameters)


# ### Decorator in Python

# Do you want to add the same block of code to different functions in Python? If so, try decorator.
# 
# In the code below, I created the decorator to track the time of the function `say_hello`.

# In[2]:


import time 

def time_func(func):
    def wrapper():
        print("This happens before the function is called")
        start = time.time()
        func()
        print('This happens after the funciton is called')
        end = time.time()
        print('The duration is', end - start, 's')

    return wrapper


# Now all I need to do is to add `@time_func` before the function `say_hello`.
# 

# In[3]:


@time_func
def say_hello():
    print("hello")

say_hello()


# Decorator makes the code clean and shortens repetitive code. If I want to track the time of another function, for example, `func2()`, I can just use:

# In[4]:


@time_func
def func2():
    pass
func2()

