#!/usr/bin/env python
# coding: utf-8

# ## Good Practices

# ### Stop using = operator to create a copy of a Python list. Use copy method instead

# When you create a copy of a Python list using the `=` operator, a change in the new list will lead to the change in the old list. It is because both lists point to the same object.

# In[1]:


l1 = [1, 2, 3]
l2 = l1 
l2.append(4)


# In[2]:


l2 


# In[3]:


l1 


# Instead of using `=` operator, use `copy()` method. Now your old list will not change when you change your new list. 

# In[4]:


l1 = [1, 2, 3]
l2 = l1.copy()
l2.append(4)


# In[5]:


l2 


# In[6]:


l1


# ### Enumerate: Get Counter and Value While Looping
# 

# Are you using `for i in range(len(array))` to access both the index and the value of the array? If so, use `enumerate` instead. It produces the same result but it is much cleaner. 

# In[7]:


arr = ['a', 'b', 'c', 'd', 'e']

# Instead of this
for i in range(len(arr)):
    print(i, arr[i])


# In[8]:


# Use this
for i, val in enumerate(arr):
    print(i, val)


# ### Difference between list append and list extend

# If you want to add a list to another list, use the `append` method. To add elements of a list to another list, use the `extend` method.

# In[9]:


# Add a list to a list
a = [1, 2, 3, 4]
a.append([5, 6])
a


# In[10]:


a = [1, 2, 3, 4]
a.extend([5, 6])

a

