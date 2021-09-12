#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Data

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

# ### DVC: A Data Version Control Tool for your Data Science Projects

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

# ### fetch_openml: Get OpenML’s Dataset in One Line of Code

# OpenML has many interesting datasets. The easiest way to get OpenML’s data in Python is to use sklearn.datasets.fetch_openml method.
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


# ### pandas-reader: Extract series data from various Internet sources directly into a pandas DataFrame

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

# ### newspaper3k: Extract Meaningful Information From an Articles in 2 Lines of Code

# If you want to quickly extract meaningful information from an article in a few lines of code, try newspaper3k. 

# In[20]:


from newspaper import Article
import nltk

nltk.download("punkt")


# In[21]:


url = "https://www.dataquest.io/blog/learn-data-science/"
article = Article(url)
article.download()
article.parse()


# In[22]:


article.title


# In[23]:


article.publish_date


# In[24]:


article.top_image


# In[25]:


article.nlp()


# In[26]:


article.summary


# In[27]:


article.keywords


# [Link to newspaper3k](https://newspaper.readthedocs.io/en/latest/user_guide/quickstart.html?highlight=parse#performing-nlp-on-an-article).
# 
# 
