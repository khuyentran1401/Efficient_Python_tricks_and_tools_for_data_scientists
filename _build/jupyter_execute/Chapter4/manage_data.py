#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Manage Data

# This section covers some tools to work with your data. 

# ### DVC: A Data Version Control Tool for Your Data Science Projects

# Git is a powerful tool to go back and forth different versions of your code. Is there a way that you can also control different versions of your data?
# 
# That is when DVC comes in handy. With DVC, you can keep the information about different versions of your data in Git while storing your original data somewhere else.
# 
# It is essentially like Git but is used for data. The code below shows how to use DVC.

# ```bash
# # Initialize
# $ dvc init
# 
# # Track data directory
# $ dvc add data # Create data.dvc
# $ git add data.dvc
# $ git commit -m "add data"
# 
# # Store the data remotely
# $ dvc remote add -d remote gdrive://lynNBbT-4J0ida0eKYQqZZbC93juUUUbVH
# 
# # Push the data to remote storage
# $ dvc push 
# 
# # Get the data
# $ dvc pull 
# 
# # Switch between different version
# $ git checkout HEAD^1 data.dvc
# $ dvc checkout
# ```

# [Link to DVC](https://dvc.org/)
# 
# Find step-by-step instructions on how to use DVC in [my article](https://towardsdatascience.com/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-7cb49c229fe0?sk=842f755cdf21a5db60aada1168c55447).

# ### sweetviz: Compare the similar features between 2 different datasets

# Sometimes it is important to compare the similar features between 2 different datasets side by side such as comparing train and test sets. If you want to quickly compare 2 datasets through graphs, check out sweetviz.

# In[17]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sweetviz as sv

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

report = sv.compare([X_train, "train data"], [X_test, "test data"])
report.show_html()


# Run the code above and you will generate a report similar to this:

# ![image](sweetviz_output.png)

# [Link to sweetviz](https://github.com/fbdesignpro/sweetviz)
