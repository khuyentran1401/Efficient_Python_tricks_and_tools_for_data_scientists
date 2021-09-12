#!/usr/bin/env python
# coding: utf-8

# ## Datetime

# ### datetime + timedelta: Calculate End DateTime Based on Start DateTime and Duration

# Provided an event starts at a certain time and takes a certain number of minutes to finish, how do you determine when it ends?
# 
# Taking the sum of `datetime` and `timedelta(minutes)` will do the trick.
# 

# In[4]:


from datetime import date, datetime, timedelta

beginning = '2020/01/03 23:59:00'
duration_in_minutes = 2500

# Find the beginning time
beginning = datetime.strptime(beginning, '%Y/%m/%d %H:%M:%S')

# Find duration in days
days = timedelta(minutes=duration_in_minutes)

# Find end time
end = beginning + days 
end


# ### Use Dates in a Month as the Feature
# 

# Have you ever wanted to use dates in a month as the feature in your time series data? You can find the days in a month by using `calendar.monthrange(year, month)[1]` like below.

# In[3]:


import calendar 

calendar.monthrange(2020, 11)[1]

