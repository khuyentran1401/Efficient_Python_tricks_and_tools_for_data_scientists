#!/usr/bin/env python
# coding: utf-8

# ## Interaction Between 2 Lists

# ### set.intersection: Find the Intersection Between 2 Sets

# If you want to get the common elements between 2 lists, convert lists to sets then use `set.intersection` to find the intersection between 2 sets.

# In[1]:


requirement1 = ['pandas', 'numpy', 'statsmodel']
requirement2 = ['numpy', 'statsmodel', 'sympy', 'matplotlib']

intersection = set.intersection(set(requirement1), set(requirement2))
list(intersection)


# ### Set Difference: Find the Difference Between 2 Sets

# If you want to find the difference between 2 lists, turn those lists into sets then apply the `difference()` method to the sets.

# In[2]:


a = [1, 2, 3, 4]
b = [1, 3, 4, 5, 6]


# In[3]:


# Find elements in a but not in b
diff = set(a).difference(set(b))
print(list(diff)) 


# In[4]:


# Find elements in b but not in a
diff = set(b).difference(set(a))
print(list(diff))  # [5, 6]

