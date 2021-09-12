#!/usr/bin/env python
# coding: utf-8

# ## Jupyter Notebook

# This section covers some tools to work with Jupyter Notebook.

# ### nbdime: Better Version Control for Jupyter Notebook

# If you want to compare the previous version and the current version of a notebook, use nbdime. The image below shows how 2 versions of a notebook are compared with nbdime.
# 
# ![image](nbdime.png)
# 
# To install nbdime, type:
# 
# ```bash
# pip install nbdime
# ```
# After installing, click the little icon in the top right corner to use nbdime.
# 
# ![image](nbdime_icon.png)
# 
# 

# [Link to nbdime](https://github.com/jupyter/nbdime/blob/master/docs/source/index.rst).

# ### display in IPython: Display Math Equations in Jupyter Notebook

# If you want to use latex to display math equations in Jupyter Notebook, use the display module in the IPython library.

# In[1]:


from IPython.display import display, Math, Latex

a = 3
b = 5
print("The equation is:")
display(Math(f'y= {a}x+{b}'))


# ### Reuse The Notebook to Run The Same Code Across Different Data

# Have you ever wanted to reuse the notebook to run the same code across different data? This could be helpful to visualize different data without changing the code in the notebook itself.
# 
# Papermill provides the tool for this. [Insert the tag `parameters` in a notebook cell that contains the variable you want to parameterize](https://papermill.readthedocs.io/en/latest/usage-parameterize.html).
# 
# Then run the code below in the terminal. 

# ```bash
# $ papermill input.ipynb output.ipynb -p data=data1
# ```

# `-p` stands for parameters. In this case, I specify the data I want to run with `-p data=<name-data>`

# [Link to papermill](https://papermill.readthedocs.io/en/latest/usage-workflow.html)

# ### watermark: Get Information About Your Hardware and the Packages Being Used within Your Notebook

# If you want to get information about your hardware and the Python packages being used within your notebook, use the magic extension watermark.
# 
# The code below shows the outputs of the watermark in my notebook.

# In[2]:


get_ipython().run_line_magic('load_ext', 'watermark')


# In[3]:


get_ipython().run_line_magic('watermark', '')


# We can also use watermark to show the versions of the libraries being used:

# In[4]:


import numpy as np 
import pandas as pd 
import sklearn


# In[5]:


get_ipython().run_line_magic('watermark', '--iversions')


# [Link to watermark](https://github.com/rasbt/watermark#installation-and-updating).
