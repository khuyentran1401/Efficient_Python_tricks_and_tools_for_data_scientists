#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Machine Learning

# ### causalimpact: Find Causal Relation of an Event and a Variable in Python	

# When working with time series data, you might want to determine whether an event has an impact on some response variable or not. For example, if your company creates an advertisement, you might want to track whether the advertisement results in an increase in sales or not.
# 
# That is when causalimpact comes in handy. causalimpact analyses the differences between expected and observed time series data. With causalimpact, you can infer the expected effect of an intervention in 3 lines of code.

# In[13]:


import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess
import causalimpact
from causalimpact import CausalImpact

# Generate random sample

np.random.seed(0)
ar = np.r_[1, 0.9]
ma = np.array([1])
arma_process = ArmaProcess(ar, ma)

X = 50 + arma_process.generate_sample(nsample=1000)
y = 1.6 * X + np.random.normal(size=1000)

# There is a change starting from index 800
y[800:] += 10


# In[14]:


data = pd.DataFrame({"y": y, "X": X}, columns=["y", "X"])
pre_period = [0, 799]
post_period = [800, 999]


# In[15]:


ci = CausalImpact(data, pre_period, post_period)
print(ci.summary())
print(ci.summary(output="report"))
ci.plot()


# ### Pipeline + GridSearchCV: Prevent Data Leakage when Scaling the Data

# Scaling the data before using GridSearchCV can lead to data leakage since the scaling tells some information about the entire data. To prevent this, assemble both the scaler and machine learning models in a pipeline then use it as the estimator for GridSearchCV. Above is an example.
# 
# The estimator is now the entire pipeline instead of just the machine learning model.

# In[16]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# load data
df = load_iris()
X = df.data
y = df.target

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Create a pipeline variable
make_pipe = make_pipeline(StandardScaler(), SVC())

# Defining parameters grid
grid_params = {"svc__C": [0.1, 1, 10, 100, 1000], "svc__gamma": [0.1, 1, 10, 100]}

# hypertuning
grid = GridSearchCV(make_pipe, grid_params, cv=5)
grid.fit(X_train, y_train)

# predict
y_pred = grid.predict(X_test)


# ### Decompose high dimensional data into two or three dimensions

# If you want to decompose high dimensional data into two or three dimensions to visualize it, what should you do? A common technique is PCA. Even though PCA is useful, I always find it complicated to create a PCA plot until I found Yellowbrick.
# 
# I really recommend using this tool if you want to visualize PCA in a few lines of code

# In[17]:


from yellowbrick.datasets import load_credit
from yellowbrick.features import PCA


# In[18]:


X, y = load_credit()
classes = ["account in defaut", "current with bills"]


# In[19]:


visualizer = PCA(scale=True, classes=classes)
visualizer.fit_transform(X, y)
visualizer.show()


# ### squared=False: Get RMSE from Sklearn’s mean_squared_error method

# If you want to get the root mean squared error using sklearn, pass `squared=False` to sklearn’s `mean_squared_error` method.

# In[42]:


from sklearn.metrics import mean_squared_error

y_actual = [1, 2, 3]
y_predicted = [1.5, 2.5, 3.5]
rmse = mean_squared_error(y_actual, y_predicted, squared=False)
rmse

