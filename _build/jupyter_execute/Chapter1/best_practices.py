#!/usr/bin/env python
# coding: utf-8

# ## Best Practices

# This section includes some best practices to write Python code. 

# ### Underscore(_): Ignore Values That Will Not Be Used

# When assigning the values returned from a function, you might want to ignore some values that are not used in future code. If so, assign those values to underscores `_`.

# In[1]:


def return_two():
    return 1, 2

_, var = return_two()
var


# ### Underscore “_”: Ignore The Index in Python For Loops

# If you want to repeat a loop a specific number of times but don’t care about the index, you can use `_`. 

# In[2]:


for _ in range(5):
    print('Hello')


# ### Python Pass Statement

# If you want to create code that does a particular thing but don’t know how to write that code yet, put that code in a function then use `pass`.
# 
# Once you have finished writing the code in a high level, start to go back to the functions and replace `pass` with the code for that function. This will prevent your thoughts from being disrupted. 

# In[3]:


def say_hello():
    pass 

def ask_to_sign_in():
    pass 

def main(is_user: bool):
    if is_user:
        say_hello()
    else:
        ask_to_sign_in()

main(is_user=True)

