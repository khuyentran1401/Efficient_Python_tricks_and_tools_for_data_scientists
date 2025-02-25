---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3.10.8 64-bit
  language: python
  name: python3
---

## Datetime

+++

![](../img/datetime.png)

+++

### datetime + timedelta: Calculate End DateTime Based on Start DateTime and Duration

+++

Provided an event starts at a certain time and takes a certain number of minutes to finish, how do you determine when it ends?

Taking the sum of `datetime` and `timedelta(minutes)` will do the trick.

```{code-cell} ipython3
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
```

### Use Dates in a Month as the Feature

+++

Have you ever wanted to use dates in a month as the feature in your time series data? You can find the days in a month by using `calendar.monthrange(year, month)[1]` like below.

```{code-cell} ipython3
import calendar 

calendar.monthrange(2020, 11)[1]
```

### Use Comparison and Arithmetic Operators on Dates in Python

+++

In Python, you can compare and subtract dates using operators.  


```{code-cell} ipython3
from datetime import date 

date1 = date(2022, 1, 1)
date2 = date(2022, 11, 1)

if date1 < date2:
    diff = date2 - date1 
else:
    diff = date1 - date2 

print(f"{date1} and {date2} is {diff} apart.")
```
