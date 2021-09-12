#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# ## Testing

# ### snoop : Smart Print to Debug Your Python Function

# If you want to figure out what is happening in your code without adding many print statements, try snoop.
# 
# To use snoop, simply add the `@snoop` decorator to a function you want to understand.

# In[2]:


import snoop 

@snoop
def factorial(x: int):
    if x == 1:
        return 1
    else: 
        return (x * factorial(x-1))


# In[3]:


num = 2
print(f'The factorial of {num} is {factorial(num)}')


# [Link to my article about snoop](https://towardsdatascience.com/3-tools-to-track-and-visualize-the-execution-of-your-python-code-666a153e435e).

# ### pytest benchmark: A Pytest Fixture to Benchmark Your Code

# If you want to benchmark your code while testing with pytest, try pytest-benchmark. 

# To understand how pytest-benchmark works, add the code below to the file named `pytest_benchmark_example.py`. Note that we add `benchmark` to the test function that we want to benchmark. 
# ```python
# # pytest_benchmark_example.py
# def list_comprehension(len_list=5):
#     return [i for i in range(len_list)]
# 
# 
# def test_concat(benchmark):
#     res = benchmark(list_comprehension)
#     assert res == [0, 1, 2, 3, 4]
# ```

# On your terminal, type:
# ```bash
# $ pytest pytest_benchmark_example.py
# ```
# Now you should see the statistics of the time it takes to excecute the test functions on your terminal:

# In[4]:


get_ipython().system('pytest pytest_benchmark_example.py')


# [Link to pytest-benchmark](https://github.com/ionelmc/pytest-benchmark).

# ### pytest.mark.parametrize: Test Your Functions with Multiple Inputs

# If you want to test your function with different examples, use `pytest.mark.parametrize` decorator.
# 
# To understand how `pytest.mark.parametrize` works, start with adding the code below to `pytest_parametrize.py`. `@pytest.mark.parametrize` is added to the test function `test_text_contain_word` to tell pytest to experiment with different pairs of inputs and expected outputs specified in its arguments. 

# ```python
# # pytest_parametrize.py
# import pytest
# 
# def text_contain_word(word: str, text: str):
#     '''Find whether the text contains a particular word'''
#     
#     return word in text
# 
# test = [
#     ('There is a duck in this text',True),
#     ('There is nothing here', False)
#     ]
# 
# @pytest.mark.parametrize('sample, expected', test)
# def test_text_contain_word(sample, expected):
# 
#     word = 'duck'
# 
#     assert text_contain_word(word, sample) == expected
# ```

# In the code above, I expect the first sentence to contain the word “duck” and expect the second sentence not to contain that word. Let's see if my expectations are correct by running:
# ```bash
# $ pytest pytest_parametrize.py
# ```

# In[5]:


get_ipython().system('pytest pytest_parametrize.py')


# Sweet! 2 tests passed when running pytest.

# [Link to my article about pytest](https://towardsdatascience.com/pytest-for-data-scientists-2990319e55e6?sk=2d3a81903b154db0c7ca832b9f29fee8).
# 
# 

# ### Pytest Fixtures: Use The Same Data for Different Tests

# Sometimes, you might want to use the same data to test different functions. If you are using pytest, use pytest fixtures to provide data to different test functions instead.

# To understand how to use pytest fixtures, start with adding the code below to the file named `pytest_fixture.py`. We use the decorator `@pytest.fixture` to create the data we want to use many times.
# ```python
# # pytest_fixture.py
# import pytest 
# from textblob import TextBlob
# 
# def extract_sentiment(text: str):
#     """Extract sentimetn using textblob. Polarity is within range [-1, 1]"""
#     
#     text = TextBlob(text)
#     return text.sentiment.polarity
# 
# @pytest.fixture 
# def example_data():
#     return 'Today I found a duck and I am happy'
# 
# def test_extract_sentiment(example_data):
#     sentiment = extract_sentiment(example_data)
#     assert sentiment > 0
# ```

# On your terminal, type:
# ```bash
# $ pytest pytest_fixture.py
# ```
# Output:

# In[6]:


get_ipython().system('pytest pytest_fixture.py ')


# ### Pytest repeat
# 

# It is a good practice to test your functions to make sure they work as expected, but sometimes you need to test 100 times until you found the rare cases when the test fails. That is when pytest-repeat comes in handy.
# 
# To understand how pytest-repeat works, start with putting the code below to the file named `pytest_repeat_example.py`. The `@pytest.mark.repeat(100)` decorator tells pytest to repeat the experiment 100 times. 

# ```python
# # pytest_repeat_example.py
# import pytest 
# import random 
# 
# def generate_numbers():
#     return random.randint(1, 100)
# 
# @pytest.mark.repeat(100)
# def test_generate_numbers():
#     assert generate_numbers() > 1 and generate_numbers() < 100
# ```

# On your terminal, type:
# ```bash
# pytest pytest_repeat_example.py
# ```
# We can see that 100 experiments are executed and passed:

# In[7]:


get_ipython().system('pytest pytest_repeat_example.py')


# [Link to pytest-repeat](https://github.com/pytest-dev/pytest-repeat)

# ### Pandera: a Python Library to Validate Your Pandas DataFrame

# The outputs of your pandas DataFrame might not be like what you expected either due to the error in your code or the change in the data format. Using data that is different from what you expected can cause errors or lead to decrease performance.
# 
# Thus, it is important to validate your data before using it. A good tool to validate pandas DataFrame is pandera. Pandera is easy to read and use.

# In[8]:


import pandera as pa
from pandera import check_input
import pandas as pd

df = pd.DataFrame({"col1": [5.0, 8.0, 10.0], "col2": ["text_1", "text_2", "text_3"]})
schema = pa.DataFrameSchema(
    {
        "col1": pa.Column(float, pa.Check(lambda minute: 5 <= minute)),
        "col2": pa.Column(str, pa.Check.str_startswith("text_")),
    }
)
validated_df = schema(df)
validated_df


# You can also use the pandera’s decorator check_inputto validates input pandas DataFrame before entering the function.

# In[9]:


@check_input(schema)
def plus_three(df):
    df["col1_plus_3"] = df["col1"] + 3
    return df


plus_three(df)


# [Link to Pandera](https://pandera.readthedocs.io/en/stable/)
