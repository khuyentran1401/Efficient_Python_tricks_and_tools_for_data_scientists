#!/usr/bin/env python
# coding: utf-8

# ## Number

# ### Get Multiples of a Number Using Modulus

# If you want to get multiples of a number, use the modulus operator `%`. The modulus operator is used to get the remainder of a division. For example, `4 % 3 = 1`, `5 % 3 = 2`.
# 
# Thus, to get multiples of `n`, we select only numbers whose remainders are 0 when dividing them by `n`. 

# In[1]:


def get_multiples_of_n(nums: list, n: int):
    """Select only numbers whose remainders
    are 0 when dividing them by n"""
    return [num for num in nums if num % n == 0]


nums = [1, 4, 9, 12, 15, 16]


# In[2]:


get_multiples_of_n(nums, 2)  # multiples of 2


# In[3]:


get_multiples_of_n(nums, 3)  # multiples of 3


# In[4]:


get_multiples_of_n(nums, 4)  # multiples of 4


# ### fractions: Get Numerical Results in Fractions instead of Decimals

# Normally, when you divide a number by another number, you will get a decimal:

# In[5]:


2 / 3 + 1


# Sometimes, you might prefer to get the results in fractions instead of decimals. There is Python built-in function called `fractions` that allows you to do exactly that. The code above shows how it works:

# In[6]:


from fractions import Fraction

res = Fraction(2 / 3 + 1)
print(res)


# Cool! We got a fraction instead of a decimal. To limit the number of decimals displayed, use `limit_denominator()`.

# In[7]:


res = res.limit_denominator()
print(res)


# What happens if we divide the result we got from `Fraction` by another number?

# In[8]:


print(res / 3)


# Nice! We got back a fraction without using the `Fraction` object again. 

# ### How to Use Underscores to Format Large Numbers in Python

# When working with a large number in Python, it can be difficult to figure out how many digits that number has. Python 3.6 and above allows you to use underscores as visual separators to group digits.
# 
# In the example below, I use underscores to group decimal numbers by thousands.
# 
# 

# In[9]:


large_num = 1_000_000
large_num


# ### Confirm whether a variable is a number
# 
# 

# If you want to confirm whether a variable is a number without caring whether it is a float or an integer, numbers. `Number` is what you can use to check.

# In[10]:


from numbers import Number

a = 2
b = 0.4

isinstance(a, Number)


# In[11]:


isinstance(b, Number)

