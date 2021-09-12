#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Feature Extraction

# ### distfit: Find The Best Theoretical Distribution For Your Data in Python

# If you want to find the best theoretical distribution for your data in Python, try `distfit`. 

# In[30]:


import numpy as np
from distfit import distfit

X = np.random.normal(0, 3, 1000)

# Initialize model
dist = distfit()

# Find best theoretical distribution for empirical data X
distribution = dist.fit_transform(X)
dist.plot()


# Besides finding the best theoretical distribution, distfit is also useful in detecting outliers. New data points that deviate significantly can then be marked as outliers.

# [Link to distfit](https://erdogant.github.io/distfit/pages/html/index.html).

# ### datefinder: Automatically Find Dates and Time in a Python String

# If you want to automatically find date and time with different formats in a Python string, try datefinder. 

# In[32]:


from datefinder import find_dates

text = """"We have one meeting on May 17th,
2021 at 9:00am and another meeting on 5/18/2021
at 10:00. I hope you can attend one of the
meetings."""

matches = find_dates(text)

for match in matches:
    print("Date and time:", match)
    print("Only day:", match.day)


# [Link to datefinder](https://github.com/akoumjian/datefinder).
# 
# 

# ### pytrends: Get the Trend of a Keyword on Google Search Over Time

# If you want to get the trend of a keyword on Google Search over time, try pytrends.
# 
# In the code below, I use pytrends to get the interest of the keyword “data science” on Google Search from 2016 to 2021.

# In[13]:


from pytrends.request import TrendReq


# In[20]:


pytrends = TrendReq(hl="en-US", tz=360)
pytrends.build_payload(kw_list=["data science"])

df = pytrends.interest_over_time()
df["data science"].plot(figsize=(20, 7))


# [Link to pytrends](https://mathdatasimplified.com/2021/04/12/pytrend-get-the-trend-of-a-keyword-on-google-search-over-time/)

# ### Fastai's add_datepart: Add Relevant DateTime Features in One Line of Code

# When working with time series, other features such as year, month, week, day of the week, day of the year, whether it is the end of the year or not, can be really helpful to predict future events. Is there a way that you can get all of those features in one line of code?
# 
# Fastai’s add_datepart method allows you to do exactly that. 

# In[41]:


import pandas as pd
from fastai.tabular.core import add_datepart
from datetime import datetime

df = pd.DataFrame(
    {
        "date": [
            datetime(2020, 2, 5),
            datetime(2020, 2, 6),
            datetime(2020, 2, 7),
            datetime(2020, 2, 8),
        ],
        "val": [1, 2, 3, 4],
    }
)

df


# In[42]:


df = add_datepart(df, "date")
df.columns


# [Link to Fastai's methods to work with tabular data](https://docs.fast.ai/tabular.core.html)

# ### Geopy: Extract Location Based on Python String

# If you work with location data, you might want to visualize them on the map. Geopy makes it easy to locate the coordinates of addresses across the globe based on a Python string.

# In[46]:


from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="find_location")
location = geolocator.geocode("30 North Circle Drive, Edwardsville, IL")


# After defining the app name and insert location, all you need to exact information about the location is to use `location.address`.

# In[47]:


location.address


# To extract the latitude and longitude or the use `location.latitide`, `location.longitude`.

# In[45]:


location.latitude, location.longitude


# [Link to Geopy](https://geopy.readthedocs.io/en/stable/)

# ### Maya: Convert the string to datetime automatically

# If you want to convert a string type to a datetime type, the common way is to use strptime(date_string, format). But it is quite inconvenient to to specify the structure of your datetime string, such as ‘ %Y-%m-%d %H:%M:%S’.
# 
# There is a tool that helps you convert the string to datetime automatically called maya. You just need to parse the string and maya will figure out the structure of your string.

# In[49]:


import maya

# Automatically parse datetime string
string = "2016-12-16 18:23:45.423992+00:00"
maya.parse(string).datetime()


# Better yet, if you want to convert the string to a different time zone (for example, CST), you can parse that into maya’s datetime function.

# In[50]:


maya.parse(string).datetime(to_timezone="US/Central")


# Check out the doc for more ways of manipulating your date string faster [here](https://github.com/timofurrer/maya).

# ### Extract holiday from date column

# You have a date column and you think the holidays might affect the target of your data. Is there an easy way to extract the holidays from the date? Yes, that is when holidays package comes in handy.
# 
# Holidays package provides a dictionary of holidays for different countries. The code below is to confirm whether 2020-07-04 is a US holiday and extract the name of the holiday.

# In[52]:


from datetime import date
import holidays

us_holidays = holidays.UnitedStates()

"2014-07-04" in us_holidays


# The great thing about this package is that you can write the date in whatever way you want and the package is still able to detect which date you are talking about.

# In[53]:


us_holidays.get("2014-7-4")


# In[54]:


us_holidays.get("2014/7/4")


# You can also add more holidays if you think that the library is lacking some holidays. Try [this](https://pypi.org/project/holidays/) out if you are looking for something similar.
# 
# 

# ### fastai’s cont_cat_split: Get a DataFrame’s Continuous and Categorical Variables Based on Their Cardinality

# To get a DataFrame’s continuous and categorical variables based on their cardinality, use fastai’s `cont_cat_split` method.
# 
# If a column consists of integers, but its cardinality is smaller than the max_card parameter, it is considered as a category variable.

# In[6]:


import pandas as pd
from fastai.tabular.core import cont_cat_split

df = pd.DataFrame(
    {
        "col1": [1, 2, 3, 4, 5],
        "col2": ["a", "b", "c", "d", "e"],
        "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
)

cont_names, cat_names = cont_cat_split(df)
print("Continuous columns:", cont_names)
print("Categorical columns:", cat_names)


# In[7]:


cont_names, cat_names = cont_cat_split(df, max_card=3)
print("Continuous columns:", cont_names)
print("Categorical columns:", cat_names)


# [Link to the documentation](https://docs.fast.ai/tabular.core.html).
