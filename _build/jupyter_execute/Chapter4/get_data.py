#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Get Data

# This section covers tools to get some data for your projects.

# ### faker: Create Fake Data in One Line of Code

# To quickly create fake data for testing, use faker.

# In[2]:


from faker import Faker

fake = Faker()

fake.color_name()


# In[3]:


fake.name()


# In[4]:


fake.address()


# In[5]:


fake.date_of_birth(minimum_age=22)


# In[6]:


fake.city()


# In[7]:


fake.job()


# [Link to faker](https://faker.readthedocs.io/en/master/)
# 
# [Link to my full article on faker](https://towardsdatascience.com/how-to-create-fake-data-with-faker-a835e5b7a9d9?sk=de199d5fdf7af9a8fd304d468e303a80).

# ### fetch_openml: Get OpenML’s Dataset in One Line of Code

# OpenML has many interesting datasets. The easiest way to get OpenML’s data in Python is to use the `sklearn.datasets.fetch_openml` method.
# 
# In one line of code, you get the OpenML’s dataset to play with!

# In[8]:


from sklearn.datasets import fetch_openml

monk = fetch_openml(name="monks-problems-2", as_frame=True)
print(monk["data"].head(10))


# ### Autoscraper

# If you want to get the data from some websites, Beautifulsoup makes it easy for you to do so. But can scraping be automated even more? If you are looking for a faster way to scrape some complicated websites such as Stackoverflow, Github in a few lines of codes, try [autoscraper](https://lnkd.in/erGHS4t).
# 
# All you need is to give it some texts so it can recognize the rule, and it will take care of the rest for you!

# In[9]:


from autoscraper import AutoScraper

url = "https://stackoverflow.com/questions/2081586/web-scraping-with-python"

wanted_list = ["How to check version of python modules?"]

scraper = AutoScraper()
result = scraper.build(url, wanted_list)

for res in result:
    print(res)


# [Link to autoscraper](https://github.com/alirezamika/autoscraper).

# ### pandas-reader: Extract Data from Various Internet Sources Directly into a Pandas DataFrame

# Have you wanted to extract series data from various Internet sources directly into a pandas DataFrame? That is when pandas_reader comes in handy.
# 
# Below is the snippet to extract daily data of AD indicator from 2008 to 2018. 

# In[10]:


import os
from datetime import datetime
import pandas_datareader.data as web

df = web.DataReader(
    "AD",
    "av-daily",
    start=datetime(2008, 1, 1),
    end=datetime(2018, 2, 28),
    api_key=os.gehide-outputtenv("ALPHAVANTAGE_API_KEY"),
)


# [Link to pandas_reader](https://pandas-datareader.readthedocs.io/en/latest/).
