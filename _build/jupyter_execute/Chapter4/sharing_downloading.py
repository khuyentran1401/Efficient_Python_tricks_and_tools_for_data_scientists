#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Sharing and Downloading

# This section covers some tools to share and download your data.

# ### Datapane: Publish your Python Objects on the Web in 2 Lines of Code

# If you want to put together your pandas.DataFrame, interactive charts such as Plotly, Bokeh, Altair, or markdown into a nice report and publish it on the web, try Datapane. The code below shows how you can publish your Python objects using Datapane in a few lines of code.

# In[6]:


import datapane as dp
import pandas as pd
import numpy as np
import plotly.express as px

# Scripts to create df and chart
df = px.data.gapminder()

chart = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60,
)

# Once you have the df and the chart, simply use
r = dp.Report(
    dp.Text("my simple report"),  # add description
    dp.DataTable(df),  # create a table
    dp.Plot(chart),  # create a chart
)

# Publish your report
r.upload(name="example")


# [Link to Datapane](https://datapane.com/)
# 
# [Link to my article about Datapane](https://towardsdatascience.com/introduction-to-datapane-a-python-library-to-build-interactive-reports-4593fd3cb9c8?sk=b8dd5203d1a37b0f08ed15ea65524d89)

# ### gdown: Download a File from Google Drive in Python

# If you want to download a file from Google Drive in Python, use gdown. All you need to specify is the URL link.

# In[10]:


import gdown

# Format of url: https://drive.google.com/uc?id=YOURFILEID
url = "https://drive.google.com/uc?id=1jI1cmxqnwsmC-vbl8dNY6b4aNBtBbKy3"
output = "Twitter.zip"

gdown.download(url, output, quiet=False)


# [Link to gdown](https://pypi.org/project/gdown/).
