#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Alternative Approach

# This section covers some alternatives approaches to work with Python. 

# ### Box: Using Dot Notation to Access Keys in a Python Dictionary
# 

# Do you wish to use `dict.key` instead of `dict['key']` to access the values inside a Python dictionary? If so, try Box.
# 
# Box is like a Python dictionary except that it allows you to access keys using dot notation. This makes the code cleaner when you want to access a key inside a nested dictionary like below.

# In[12]:


from box import Box

food_box = Box({"food": {"fruit": {"name": "apple", "flavor": "sweet"}}})
print(food_box)


# In[13]:


print(food_box.food.fruit.name)


# [Link to Box](https://github.com/cdgriffith/Box).

# ### decorator module: Write Shorter Python Decorators without Nested Functions

# Have you ever wished to write a Python decorator with only one function instead of nested functions like below?
# 

# In[15]:


from time import time, sleep


def time_func_complex(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        func(*args, **kwargs)
        end_time = time()
        print(
            f"""It takes {round(end_time - start_time, 3)} seconds to execute the function"""
        )

    return wrapper


@time_func_complex
def test_func_complex():
    sleep(1)


test_func_complex()


# If so, try decorator. In the code below, `time_func_simple` produces the exact same results as `time_func_complex`, but `time_func_simple` is easier and short to write.
# 

# In[17]:


from decorator import decorator


@decorator
def time_func_simple(func, *args, **kwargs):
    start_time = time()
    func(*args, **kwargs)
    end_time = time()
    print(
        f"""It takes {round(end_time - start_time, 3)} seconds to execute the function"""
    )


@time_func_simple
def test_func_simple():
    sleep(1)


test_func()


# [Check out other things the decorator library can do](https://github.com/micheles/decorator).

# ### virtualenv-clone: Create a Copy of a Virtual Environment

# Sometimes you might want to use the same virtual environment for 2 different directories. If you want to create a copy of a virtual environment, use virtualenv-clone. 
# 
# The code below shows how to use virtualenv-clone.

# ```bash
# $ pip install virtualenv-clone
# $ virtualenv-clone old_venv/ new_venv/
# 
# $ source new_venv/bin/activate
# ```

# [Link to virtualenv-clone](https://github.com/edwardgeorge/virtualenv-clone).
