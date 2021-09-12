#!/usr/bin/env python
# coding: utf-8

# ## Visualization

# This section covers some tools to visualize your data and model.

# ### Graphviz: Create a Flowchart to Capture Your Ideas in Python

# A flowchart is helpful for summarizing and visualizing your workflow. This also helps your team understand your workflow. Wouldn’t it be nice if you could create a flowchart using Python?
# 
# Graphviz makes it easy to create a flowchart like below. 

# In[2]:


from graphviz import Graph 

# Instantiate a new Graph object
dot = Graph('Data Science Process', format='png')

# Add nodes
dot.node('A', 'Get Data')
dot.node('B', 'Clean, Prepare, & Manipulate Data')
dot.node('C', 'Train Model')
dot.node('D', 'Test Data')
dot.node('E', 'Improve')

# Connect these nodes
dot.edges(['AB', 'BC', 'CD', 'DE'])

# Save chart
dot.render('data_science_flowchart', view=True)


# In[3]:


dot 


# [Link to graphviz](https://graphviz.readthedocs.io/en/stable/manual.html)

# ### Create an Interactive Map in Python
# 

# If you want to create a map provided the location in a few lines of code, try folium. Folium is a Python library that allows you to create an interactive map.

# In[5]:


import folium
m = folium.Map(location=[45.5236, -122.6750])

tooltip = 'Click me!'
folium.Marker([45.3288, -121.6625], popup='<i>Mt. Hood Meadows</i>',
              tooltip=tooltip).add_to(m)
m 


# View the document of folium [here](https://python-visualization.github.io/folium/quickstart.html#Getting-Started).
# 
# I used this library to [view the locations of the owners of top machine learning repositories](https://towardsdatascience.com/top-6-python-libraries-for-visualization-which-one-to-use-fe43381cd658#4e40). Pretty cool to see their locations through an interactive map.

# ### dtreeviz: Visualize and Interpret a Decision Tree Model

# If you want to find an easy way to visualize and interpret a decision tree model, use dtreeviz.

# In[7]:



from dtreeviz.trees import dtreeviz
from sklearn import tree
from sklearn.datasets import load_wine

wine = load_wine()
classifier = tree.DecisionTreeClassifier(max_depth=2)
classifier.fit(wine.data, wine.target)

vis = dtreeviz(
    classifier,
    wine.data,
    wine.target,
    target_name="wine_type",
    feature_names=wine.feature_names,
)

vis.view()


# The image below shows the output of dtreeviz when applying it on DecisionTreeClassifier.

# ![image](DTreeViz.png)

# [Link to dtreeviz](https://github.com/parrt/dtreeviz).
