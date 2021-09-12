#!/usr/bin/env python
# coding: utf-8

# ## Tools for Best Python Practices

# This section cover tools that encourage best Python practices. 

# ### Don’t Hard-Code. Use Hydra Instead

# When writing code, it is a good practice to put the values that you might change in a separate file from your original script.
# 
# This practice not only saves you from wasting time searching for a specific variable in your scripts but also makes your scripts more reproducible.
# 
# My favorite tool to handle config files is Hydra. The code below shows how to get values from a config file using Hydra.

# All parameters are specified in a configuration file named `config.yaml`: 
# 
# ```yaml
# # config.yaml
# data: data1 
# variables: 
#   drop_features: ['iid', 'id', 'idg', 'wave']
#   categorical_vars: ['undergra', 'zipcode']
#  ```

# In seperate file named `main.py`, the parameters in the `config.yaml` file are called using Hydra:
# ```python
# # main.py
# import hydra 
# 
# @hydra.main(config_name='config.yaml')
# def main(config):
#     print(f'Process {config.data}')
#     print(f'Drop features: {config.variables.drop_features}')
# 
# if __name__ == '__main__':
#     main()
# ```

# On your terminal, type:
# ```bash
# $ python main.py
# ```
# Output:

# In[1]:


get_ipython().system('python hydra_examples/main.py')


# [Link to my article about Hydra](https://towardsdatascience.com/introduction-to-hydra-cc-a-powerful-framework-to-configure-your-data-science-projects-ed65713a53c6?sk=eb08126922cc54a40c2fdfaea54c708d).
# 
# [Link to Hydra](https://hydra.cc/). 
# 
# 

# ### python-dotenv: How to Load the Secret Information from .env File

# An alternative to saving your secret information to the environment variable is to save it to `.env` file in the same path as the root of your project.  
# 

# ```bash
# # .env
# USERNAME=my_user_name
# PASSWORD=secret_password
# ```

# The easiest way to load the environment variables from `.env` file is to use python-dotenv library.

# In[2]:


from dotenv import load_dotenv
import os 

load_dotenv()
PASSWORD = os.getenv('PASSWORD')
print(PASSWORD)


# [Link to python-dotenv](https://github.com/theskumar/python-dotenv)

# ### kedro Pipeline: Create Pipeline for Your Data Science Projects in Python

# When writing code for a data science project, it can be difficult to understand the workflow of the code. Is there a way that you can create different components based on their functions then combine them together in one pipeline?
# 
# That is when kedro comes in handy. In the code below, each node is one component of the pipeline. We can specify the input and output of the node. Then combine them together using `Pipeline`.
# 
# `DataCatalog` is the input data. Structuring your code this way makes it easier for you and others to follow your code logic.

# In[3]:


from kedro.pipeline import node, Pipeline
from kedro.io import DataCatalog, MemoryDataSet
from kedro.runner import SequentialRunner

# Prepare a data catalog
data_catalog = DataCatalog({"data.csv": MemoryDataSet()})

# Prepare first node
def process_data():
    return f"processed data"

process_data_node = node(
    func=process_data, inputs=None, outputs="processed_data"
)

def train_model(data: str):
    return f"Training model using {data}"

train_model_node = node(
    func=train_model, inputs="processed_data", outputs="trained_model"
)

# Assemble nodes into a pipeline
pipeline = Pipeline([process_data_node, train_model_node])


# In[4]:


# Create a runner to run the pipeline
runner = SequentialRunner()
print(runner.run(pipeline, data_catalog))


# [Link to my article about Kedro](https://towardsdatascience.com/kedro-a-python-framework-for-reproducible-data-science-project-4d44977d4f04)
# 
# [Link to Kedro](https://kedro.readthedocs.io/en/stable/02_get_started/03_hello_kedro.html)

# ### docopt: Create Beautiful Command-line Interfaces for Documentation in Python

# Writing documentation for your Python script helps others understand how to use your script. However, instead of making them spend some time to find the documentation in your script, wouldn’t it be nice if they can view the documentation in the terminal?
# 
# That is when docopt comes in handy. docopt allows you to create beautiful command-line interfaces by passing a Python string. 

# To understand how docopt works, we can add a docstring at the beginning of the file named `docopt_example.py`. 
# ```python
# # docopt_example.py
# """Extract keywords of an input file
# Usage:
#     docopt_example.py --data-dir=<data-directory> [--input-path=<path>]
# Options:
#     --data-dir=<path>    Directory of the data
#     --input-path=<path>  Name of the input file [default: input_text.txt]
# """
# 
# from docopt import docopt 
# 
# if __name__ == '__main__':
#     args = docopt(__doc__, argv=None, help=True)
#     data_dir = args['--data-dir']
#     input_path = args['--input-path']
# 
#     if data_dir:
#         print(f"Extracting keywords from {data_dir}/{input_path}")
# ```

# Running the file `docopt_example.py` should give us the output like below:
# 
# ```bash
# $ python docopt_example.py
# ```

# In[5]:


get_ipython().system('python docopt_example.py')


# [Link to docopt](http://docopt.org/).
