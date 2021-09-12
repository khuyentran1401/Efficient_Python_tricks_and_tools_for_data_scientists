#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## String

# ### String find: Find The Index of a Substring in a Python String

# If you want to find the index of a substring in a string, use `find()` method. This method will return the index of the first occurrence of the substring if found and return `-1` otherwise.

# In[17]:


sentence = "Today is Saturaday"

# Find the index of first occurrence of the substring
sentence.find("day")


# In[19]:


sentence.find("nice")
# No substring is found


# You can also provide the starting and stopping position of the search:

# In[18]:


# Start searching for the substring at index 3
sentence.find("day", 3)


# ### re.sub: Replace One String with Another String Using Regular Expression	

# If you want to either replace one string with another string or to change the order of characters in a string, use `re.sub`.
# 
# `re.sub` allows you to use a regular expression to specify the pattern of the string you want to swap.
# 
# In the code below, I replace `3/7/2021` with `Sunday` and replace `3/7/2021` with `2021/3/7`.

# In[1]:


import re

text = "Today is 3/7/2021"
match_pattern = r"(\d+)/(\d+)/(\d+)"

re.sub(match_pattern, "Sunday", text)


# In[2]:


re.sub(match_pattern, r"\3-\1-\2", text)

