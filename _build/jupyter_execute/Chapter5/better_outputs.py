#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Better Outputs

# ### How to Strip Outputs and Execute Interactive Code in a Python Script
# 

# Have you ever seen a tutorial with an interactive Python code and wished to execute it in a Python script like above?
# 
# It might be time-consuming to delete all `>>>` symbols and remove all outputs, especially when the code is long. That is why I created strip-interactive.

# In[3]:


from strip_interactive import run_interactive

code = """
>>> import numpy as np
>>> print(np.array([1,2,3]))
[1 2 3]
>>> print(np.array([4,5,6]))
[4 5 6]
"""

clean_code = run_interactive(code)


# [Link to the article about strip-interactive](https://towardsdatascience.com/how-to-strip-outputs-and-execute-interactive-code-in-a-python-script-6d4c5da3beb0?sk=1db3d887884ad2429b9c78e1c72a2a4d).
# 
# [Link to strip-interactive](https://github.com/khuyentran1401/strip_interactive).

# ### rich.inspect: Produce a Beautiful Report on any Python Object

# If you want to quickly see which attributes and methods of a Python object are available, use rich’s `inspect` method.
# 
# rich’s `inspect` method allows you to create a beautiful report for any Python object, including a string.

# In[6]:


from rich import inspect

print(inspect('hello', methods=True))


# ### Rich’s Console: Debug your Python Function in One Line of Code

# Sometimes, you might want to know which elements in the function created a certain output. Instead of printing every variable in the function, you can simply use Rich’s `Console` object to print both the output and all the variables in the function.

# In[7]:


from rich import console
from rich.console import Console 
import pandas as pd 

console = Console()

data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

def edit_data(data):
    var_1 = 45
    var_2 = 30
    var_3 = var_1 + var_2
    data['a'] = [var_1, var_2, var_3]
    console.log(data, log_locals=True)

edit_data(data)


# [Link to my article about rich](https://towardsdatascience.com/rich-generate-rich-and-beautiful-text-in-the-terminal-with-python-541f39abf32e).
# 
# [Link to rich](https://github.com/willmcgugan/rich).

# ### loguru: Print Readable Traceback in Python
# 

# Sometimes, it is difficult to understand the traceback and to know which inputs cause the error. Is there a way that you can print a more readable traceback?
# 
# That is when loguru comes in handy. By adding decorator `logger.catch` to a function, loguru logger will print a more readable trackback and save the traceback to a separate file  like below
# 

# In[9]:


from sklearn.metrics import mean_squared_error
import numpy as np
from loguru import logger

logger.add("file_{time}.log", format="{time} {level} {message}")

@logger.catch
def evaluate_result(y_true: np.array, y_pred: np.array):
    mean_square_err = mean_squared_error(y_true, y_pred)
    root_mean_square_err = mean_square_err ** 0.5

y_true = np.array([1, 2, 3])
y_pred = np.array([1.5, 2.2])
evaluate_result(y_true, y_pred)


# ```bash
# > File "/tmp/ipykernel_174022/1865479429.py", line 14, in <module>
#     evaluate_result(y_true, y_pred)
#     │               │       └ array([1.5, 2.2])
#     │               └ array([1, 2, 3])
#     └ <function evaluate_result at 0x7f279588f430>
# 
#   File "/tmp/ipykernel_174022/1865479429.py", line 9, in evaluate_result
#     mean_square_err = mean_squared_error(y_true, y_pred)
#                       │                  │       └ array([1.5, 2.2])
#                       │                  └ array([1, 2, 3])
#                       └ <function mean_squared_error at 0x7f27958bfca0>
# 
#   File "/home/khuyen/book/venv/lib/python3.8/site-packages/sklearn/utils/validation.py", line 63, in inner_f
#     return f(*args, **kwargs)
#            │  │       └ {}
#            │  └ (array([1, 2, 3]), array([1.5, 2.2]))
#            └ <function mean_squared_error at 0x7f27958bfb80>
#   File "/home/khuyen/book/venv/lib/python3.8/site-packages/sklearn/metrics/_regression.py", line 335, in mean_squared_error
#     y_type, y_true, y_pred, multioutput = _check_reg_targets(
#             │       │                     └ <function _check_reg_targets at 0x7f27958b7af0>
#             │       └ array([1.5, 2.2])
#             └ array([1, 2, 3])
#   File "/home/khuyen/book/venv/lib/python3.8/site-packages/sklearn/metrics/_regression.py", line 88, in _check_reg_targets
#     check_consistent_length(y_true, y_pred)
#     │                       │       └ array([1.5, 2.2])
#     │                       └ array([1, 2, 3])
#     └ <function check_consistent_length at 0x7f279676e040>
#   File "/home/khuyen/book/venv/lib/python3.8/site-packages/sklearn/utils/validation.py", line 319, in check_consistent_length
#     raise ValueError("Found input variables with inconsistent numbers of"
# 
# ValueError: Found input variables with inconsistent numbers of samples: [3, 2]
# ```

# [Link to loguru](https://github.com/Delgan/loguru). 

# ### Icrecream: Never use print() to debug again

# If you use print or log to debug your code, you might be confused about which line of code creates the output, especially when there are many outputs.
# 
# You might insert text to make it less confusing, but it is time-consuming.
# 
# Try icecream instead. Icrecream inspects itself and prints both its own arguments and the values of those arguments like below.

# In[14]:


from icecream import ic

def plus_one(num):
    return num + 1

# Instead of this
print('output of plus_on with num = 1:', plus_one(1))
print('output of plus_on with num = 2:', plus_one(2))

# Use this
ic(plus_one(1))
ic(plus_one(2))


# [Link to icecream](https://github.com/gruns/icecream)
# 
# [Link to my article about icecream](https://towardsdatascience.com/stop-using-print-to-debug-in-python-use-icecream-instead-79e17b963fcc)

# ### Pyfiglet: Make Large and Unique Letters Out of Ordinary Text in Python

# If you want to make large and unique letters out of ordinary text using Python, try pyfiglet. Below are some outputs of pyfiglet:

# In[18]:


import pyfiglet
from termcolor import colored, cprint

out = pyfiglet.figlet_format("Hello")
print(out)


# In[20]:


out = pyfiglet.figlet_format("Hello", font='slant')
print(out)


# In[21]:


cprint(pyfiglet.figlet_format('Hello', font='bell'), 'blue')


# This could be used as the welcome message for your Python package 🙂

# [Link to pyfiglet](https://github.com/pwaller/pyfiglet).
# 
# [Link to termcolor](https://pypi.org/project/termcolor/).

# ### heartrate — Visualize the Execution of a Python Program in Real-Time

# If you want to visualize which lines are executed and how many times they are executed, try heartrate.
# 
# You only need to add two lines of code to use heartrate.

# In[24]:


import heartrate 
heartrate.trace(browser=True)

def factorial(x):
    if x == 1:
        return 1
    else:
        return (x * factorial(x-1))


if __name__ == "__main__":
    num = 5
    print(f"The factorial of {num} is {factorial(num)}")


# You should see something similar to the below when opening the browser:
# 
# ![image](heartrate.png)

# [Link to heartrate](https://github.com/alexmojaki/heartrate).
