#!/usr/bin/env python
# coding: utf-8

# ## Automation

# This section covers some tools to automate your Python code.

# ### Schedule: Schedule your Python Functions to Run At a Specific Time

# If you want to schedule Python functions to run periodically at a certain day or time of the week, use schedule.
# 
# In the code snippet below, I use schedule to get incoming data at 10:30 every day and train the model at 8:00 every Wednesday.
# 
# ```python
# import schedule 
# import time 
# 
# def get_incoming_data():
#     print("Get incoming data")
# 
# def train_model():
#     print("Retraining model")
# 
# schedule.every().day.at("10:30").do(get_incoming_data)
# schedule.every().wednesday.at("08:00").do(train_model)
# 
# while True:
#     schedule.run_pending()
#     time.sleep(1)
# ```

# [Link to schedule](https://github.com/dbader/schedule)

# ### notify-send: Send a Desktop Notification after Finishing Executing a File

# If you want to receive a desktop notification after finishing executing a file in Linux, use notify-send.
# 
# In the code below, after finishing executing `file_to_run.py`, you will receive a notification on the top of your screen to inform you that the process is terminated.
# 
# ```bash
# python file_to_run.py ; notify-send "Process terminated"
# ```

# ### isort: Automatically Sort your Python Imports in 1 Line of Code

# When your code grows bigger, you might need to import a lot of libraries, and it can be confusing to look at. Instead of manually organing your imports, use isort.
# 
# isort is a Python library to sort imports alphabetically and automatically separated into sections and by type. You just need to use `isort name_of_your_file.py` to sort your imports.
# 
# Below is how the imports look like before sorting.
# 
# 

# ```python
# from sklearn.metrics import confusion_matrix, fl_score, classification_report, roc_curve
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# from sklearn import svm
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import TimeSeriesSplit
# ```
# 
# On your terminal, type:
# ```bash
# isort name_of_your_file.py
# ```
# 
# Now the imports are much more organized!
# ```python
# from sklearn import svm
# from sklearn.metrics import (classification_report, confusion_matrix, fl_score,
#                              roc_curve)
# from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
#                                      TimeSeriesSplit, train_test_split)
# from sklearn.naive_bayes import GaussianNB, MultinomialNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# ```

# [Link to isort](https://github.com/pycqa/isort).

# ### knockknock: Receive an Email When Your Code Finishes Executing

# It can take hours or days to train a model and you can be away from the computer when your model finishes training. Wouldn’t it be nice to receive an email when your code finishes executing? There is an app for that knock-knock.
# 
# All it takes is one line of code specifying your email address.

# ```python
# from knockknock import email_sender 
# 
# @email_sender(recipient_emails=['<your_email@address.com>', '<your_second_email@adress.com>'],
# sender_email="<grandma's_email@gmail.com>")
# def train_your_nicest_model(your_nicest_parameters):
#     import time 
#     time.sleep(10_000)
#     return {'loss': 0.9}
# ```

# You can even have it send to your slack channel so everybody in your team can see. See the docs of this library [here](https://github.com/huggingface/knockknock).

# ### snscrape: Scrape Social Networking Services in Python

# If you want to scrape social networking services such as Twitter, Facebook, Reddit, etc, try snscrape.
# 
# For example, you can use snsscrape to scrape all tweets from a user or get the latest 100 tweets with the hashtag #python.

# ```bash
# # Scrape all tweets from @KhuyenTran16
# snscrape twitter-user KhuyenTran16
# 
# # Save outputs
# snscrape twitter-user KhuyenTran16 >> khuyen_tweets 
# 
# # Scrape 100 tweets with hashtag python
# snscrape --max-results 100 twitter-hashtag python
# ```

# [Link to snscrape](https://github.com/JustAnotherArchivist/snscrape).

# ### Typer: Build a Command-Line Interface in a Few Lines of Code
# 

# The last thing you want to happen is to have users dig into your code to run it. Is there a way that users can insert arguments into your code on the command line?
# 
# That is when Typer comes in handy. Typer allows you to build a command-line interface in a few lines of code based on Python-type hints.

# For example, in a file named `typer_example`, write:

# ```python
# import typer 
# 
# def process_data(data: str, version: int):
#     print(f'Processing {data},' 
#           f'version {version}')
# 
# if __name__ == '__main__':
#     typer.run(process_data)
# ```

# On your terminal, type:
# ```bash
# python typer_example.py data 1
# ```
# And you should see an output like below:

# In[20]:


get_ipython().system('python typer_example.py data 1')


# [Link to Typer](https://typer.tiangolo.com/).
# 
# [My full article about Typer](https://towardsdatascience.com/typer-build-powerful-clis-in-one-line-of-code-using-python-321d9aef3be8).

# ### yarl: Create and Extract Elements from a URL Using Python

# If you want to easily create and extract elements from a URL using Python, try yarl. In the code below, I use yarl to extract different elements of the URL https://github.com/search?q=data+science.

# In[23]:


from yarl import URL 

url = URL('https://github.com')
new_url = url/ "search" % 'q=data+science'
print(new_url) 


# In[24]:


print(new_url.host) 


# In[25]:


print(new_url.path) 


# In[26]:


print(new_url.query_string) 


# [Link to yarl](https://github.com/aio-libs/yarl).

# ### interrogate: Check your Python Code for Missing Docstrings

# Sometimes, you might forget to write docstrings for classes and functions. Instead of manually looking at all your functions and classes for missing docstrings, use interrogate instead.
# 
# In the code below, I use interrogate to check for missing docstrings in the file `interrogate_example.py`.

# ```python
# # interrogate_example.py
# class Math:
#     def __init__(self, num) -> None:
#         self.num = num
# 
#     def plus_two(self):
#         """Add 2"""
#         return self.num + 2
# 
#     def multiply_three(self):
#         return self.num * 3
# ```

# On your terminal, type:
# ```bash
# interrogate interrogate_example.py
# ```
# Output:

# In[29]:


get_ipython().system('interrogate interrogate_example.py')


# To automatically check for missing docstrings whenever you commit new files, add interrogate to your pre-commit hooks. [Here](https://towardsdatascience.com/4-pre-commit-plugins-to-automate-code-reviewing-and-formatting-in-python-c80c6d2e9f5#1bec) is how to do that.

# [Link to interrogate](https://interrogate.readthedocs.io/en/latest/).

# ### mypy: Static Type Checker for Python
# 

# Type hinting in Python is useful for other developers to understand which data types to insert into your function. To automatically type check your code, use mypy. 
# 
# To see how mypy works, start with writing a normal code that uses type hinting. We name this file `mypy_example.py`

# ```python
# # mypy_example.py
# from typing import List, Union
# 
# def get_name_price(fruits: list) -> Union[list, tuple]:
#     return zip(*fruits)
# 
# fruits = [('apple', 2), ('orange', 3), ('grape', 2)]
# names, prices = get_name_price(fruits)
# print(names)  # ('apple', 'orange', 'grape')
# print(prices)  # (2, 3, 2)
# ```

# On your terminal, type:
# ```bash
# mypy mypy_example.py
# ```
# And you should see something like below:

# In[31]:


get_ipython().system('mypy mypy_example.py')


# [Link to mypy](https://mypy.readthedocs.io/en/latest/introduction.html).
