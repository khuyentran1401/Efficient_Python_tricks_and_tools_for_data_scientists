#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# # Python Built-in Methods

# ## Number

# ### Get Multiples of a Number Using Modulus

# If you want to get multiples of a number, use the modulus operator `%`. The modulus operator is used to get the remainder of a division. For example, `4 % 3 = 1`, `5 % 3 = 2`.
# 
# Thus, to get multiples of `n`, we select only numbers whose remainders are 0 when dividing them by `n`. 

# In[2]:


def get_multiples_of_n(nums: list, n: int):
    """Select only numbers whose remainders
    are 0 when dividing them by n"""
    return [num for num in nums if num % n == 0]


nums = [1, 4, 9, 12, 15, 16]


# In[3]:


get_multiples_of_n(nums, 2)  # multiples of 2


# In[4]:


get_multiples_of_n(nums, 3)  # multiples of 3


# In[5]:


get_multiples_of_n(nums, 4)  # multiples of 4


# ### fractions: Get Numerical Results in Fractions instead of Decimals

# Normally, when you divide a number by another number, you will get a decimal:

# In[6]:


2 / 3 + 1


# Sometimes, you might prefer to get the results in fractions instead of decimals. There is Python built-in function called fractions that allows you to do exactly that. The code above shows how it works:

# In[7]:


from fractions import Fraction

res = Fraction(2 / 3 + 1)
print(res)


# Cool! We got a fraction instead of a decimal. To limit the number of decimals displayed, use `limit_denominator()`.

# In[8]:


res = res.limit_denominator()
print(res)


# What happens if we divide the result we got from `Fraction` by another number?

# In[9]:


print(res / 3)


# Nice! We got back a fraction without using the `Fraction` object again. 

# Being able to turn decimals into fractions is nice, but what if want to get other nice math outputs in Python such as $\sqrt{8} = 2\sqrt{2}$? That is when SymPy comes in handy.

# ### How to Use Underscores to Format Large Numbers in Python

# In[10]:


large_num = 1_000_000
large_num


# ### Confirm whether a variable is a number
# 
# 

# In[11]:


from numbers import Number

a = 2
b = 0.4

isinstance(a, Number)


# In[12]:


isinstance(b, Number)


# ## Boolean Operators: Connect Two Boolean Expressions into One Expression

# In[13]:


movie_available = True
have_money = False

get_excited = movie_available | have_money
get_excited


# In[14]:


buy = movie_available & have_money
buy


# ## String

# ### String find: Find the Index of a Substring in a Python STring

# In[15]:


sentence = "Today is Saturaday"

# Find the index of first occurrence of the substring
sentence.find("day")


# In[16]:


# Start searching for the substring at index 3
sentence.find("day", 3)


# In[17]:


sentence.find("nice")
# No substring is found


# ### re.sub: Replace One String with Another String Using Regular Expression	

# In[18]:


import re

text = "Today is 3/7/2021"
match_pattern = r"(\d+)/(\d+)/(\d+)"

re.sub(match_pattern, "Sunday", text)


# In[19]:


re.sub(match_pattern, r"\3-\1-\2", text)


# ## List

# ### any: Check if Any Element of an Iterable is True

# In[20]:


text = "abcdE"
any(c for c in text if c.isupper())


# ### Extended Iterable Unpacking: Ignore Multiple Values when Unpacking a Python Iterable

# In[21]:


a, *_, b = [1, 2, 3, 4]
print(a)


# In[22]:


b


# In[23]:


_


# ### How to Unpack Iterables in Python	

# In[24]:


nested_arr = [[1, 2, 3], ["a", "b"], 4]
num_arr, char_arr, num = nested_arr


# In[25]:


num_arr


# In[26]:


char_arr


# ## Class

# ### __str__ and __repr__: Create a String Representation of a Python Object
# 

# In[27]:


class Food:
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color

    def __str__(self):
        return f"{self.color} {self.name}"

    def __repr__(self):
        return f"Food({self.color}, {self.name})"


food = Food("apple", "red")

print(food)  #  str__


# In[28]:


food  # __repr__

