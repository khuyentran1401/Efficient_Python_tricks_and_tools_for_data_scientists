#!/usr/bin/env python
# coding: utf-8

# ## General

# ### join method: Turn an Iterable into a Python String

# If you want to turn an iterable into a string, use `join()`.
# 
# In the code below, I join elements in the list fruits using “, “.

# In[1]:


fruits = ['apples', 'oranges', 'grapes']

fruits_str = ', '.join(fruits)

print(f"Today, I need to get some {fruits_str} in the grocery store")


# ### Zip: Associate Elements from Two Iterators based on the Order

# If you want to associate elements from two iterators based on the order, combine `list` and `zip`. 

# In[2]:


nums = [1, 2, 3, 4]
string = "abcd"
combinations = list(zip(nums, string))
for comb in combinations:
    print(comb)


# ### Zip Function: Create Pairs of Elements from Two Lists in Python

# If you want to create pairs of elements from two lists, use `zip`. `zip()` function takes iterables and aggregates them in a tuple.
# 
# You can also unzip the list of tuples by using `zip(*list_of_tuples)`.
# 

# In[3]:


nums = [1, 2, 3, 4]
chars = ['a', 'b', 'c', 'd']

comb = list(zip(nums, chars))
comb 


# In[4]:


nums_2, chars_2 = zip(*comb)
nums_2, chars_2


# ### Stop using = operator to create a copy of a Python list. Use copy method instead

# When you create a copy of a Python list using the `=` operator, a change in the new list will lead to the change in the old list. It is because both lists point to the same object.

# In[5]:


l1 = [1, 2, 3]
l2 = l1 
l2.append(4)


# In[6]:


l2 


# In[7]:


l1 


# Instead of using `=` operator, use `copy()` method. Now your old list will not change when you change your new list. 

# In[8]:


l1 = [1, 2, 3]
l2 = l1.copy()
l2.append(4)


# In[9]:


l2 


# In[10]:


l1


# ### Enumerate: Get Counter and Value While Looping
# 

# Are you using `for i in range(len(array))` to access both the index and the value of the array? If so, use `enumerate` instead. It produces the same result but it is much cleaner. 

# In[11]:


arr = ['a', 'b', 'c', 'd', 'e']

# Instead of this
for i in range(len(arr)):
    print(i, arr[i])


# In[12]:


# Use this
for i, val in enumerate(arr):
    print(i, val)


# ### set.intersection: Find the Intersection Between 2 Sets

# If you want to get the common elements between 2 lists, convert lists to sets then use `set.intersection` to find the intersection between 2 sets.

# In[13]:


requirement1 = ['pandas', 'numpy', 'statsmodel']
requirement2 = ['numpy', 'statsmodel', 'sympy', 'matplotlib']

intersection = set.intersection(set(requirement1), set(requirement2))
list(intersection)


# ### Set Difference: Find the Difference Between 2 Sets

# If you want to find the difference between 2 lists, turn those lists into sets then apply the `difference()` method to the sets.

# In[14]:


a = [1, 2, 3, 4]
b = [1, 3, 4, 5, 6]


# In[15]:


# Find elements in a but not in b
diff = set(a).difference(set(b))
print(list(diff)) 


# In[16]:


# Find elements in b but not in a
diff = set(b).difference(set(a))
print(list(diff))  # [5, 6]


# ### Difference between list append and list extend

# If you want to add a list to another list, use the `append` method. To add elements of a list to another list, use the `extend` method.

# In[17]:


# Add a list to a list
a = [1, 2, 3, 4]
a.append([5, 6])
a


# In[18]:


a = [1, 2, 3, 4]
a.extend([5, 6])

a


# In[ ]:




