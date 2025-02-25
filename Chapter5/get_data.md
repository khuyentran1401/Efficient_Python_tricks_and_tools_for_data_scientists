---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Get Data

+++

This section covers tools to get some data for your projects.

+++

### faker: Create Fake Data in One Line of Code

```{code-cell} ipython3
:tags: [hide-cell]

!pip install Faker
```

For quickly generating fake data for testing, use `Faker`.

Here are a few examples:

```{code-cell} ipython3
from faker import Faker

fake = Faker()

fake.color_name()
```

```{code-cell} ipython3
fake.name()
```

```{code-cell} ipython3
fake.address()
```

```{code-cell} ipython3
fake.date_of_birth(minimum_age=22)
```

```{code-cell} ipython3
fake.city()
```

```{code-cell} ipython3
fake.job()
```

[Link to faker](https://faker.readthedocs.io/en/master/)

+++

### Silly: Produce Silly Test Data

```{code-cell} ipython3
:tags: [hide-cell]

!pip install silly
```

For generating playful test data, try the `silly` library.

Here are some examples:


```{code-cell} ipython3
import silly 

name = silly.name()
email = silly.email()
print(f"Her name is {name}. Her email is {email}")
```

```{code-cell} ipython3
silly.a_thing()
```

```{code-cell} ipython3
silly.thing()
```

```{code-cell} ipython3
silly.things()
```

```{code-cell} ipython3
silly.sentence()
```

```{code-cell} ipython3
silly.paragraph()
```

[Link to silly](https://github.com/cube-drone/silly).

+++

### Random User: Generate Random User Data in One Line of Code

+++

Need to generate fake user data for testing? The Random User Generator API provides a simple way to get random user data. Here’s how you can fetch and use this data in your code:

```{code-cell} ipython3
import json
from urllib.request import urlopen

# Show 2 random users
data = urlopen("https://randomuser.me/api?results=2").read()
users = json.loads(data)["results"]
users
```

[Link to Random User Generator](https://randomuser.me/).

+++

### fetch_openml: Get OpenML’s Dataset in One Line of Code

+++

OpenML offers a variety of intriguing datasets. You can easily fetch these datasets in Python using `sklearn.datasets.fetch_openml`.

Here’s how to load an OpenML dataset with a single line of code:

```{code-cell} ipython3
from sklearn.datasets import fetch_openml

monk = fetch_openml(name="monks-problems-2", as_frame=True)
print(monk["data"].head(10))
```

### Autoscraper: Automate Web Scraping in Python

```{code-cell} ipython3
:tags: [hide-cell]

!pip install autoscraper
```

To automate web scraping with minimal code, try `autoscraper`.

Here's a quick example to extract specific elements from a webpage:

```{code-cell} ipython3
from autoscraper import AutoScraper

url = "https://stackoverflow.com/questions/2081586/web-scraping-with-python"

wanted_list = ["How to check version of python modules?"]

scraper = AutoScraper()
result = scraper.build(url, wanted_list)

for res in result:
    print(res)
```

[Link to autoscraper](https://github.com/alirezamika/autoscraper).

+++

### pandas-reader: Extract Data from Various Internet Sources Directly into a Pandas DataFrame

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pandas-datareader
```

To retrieve time series data from various internet sources directly into a pandas DataFrame, use `pandas-datareader`.

Here’s how you can fetch daily data of the AD indicator from 2008 to 2018:

```{code-cell} ipython3
:tags: [hide-output]

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
```

[Link to pandas_reader](https://pandas-datareader.readthedocs.io/en/latest/).

+++

### pytrends: Get the Trend of a Keyword on Google Search Over Time

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pytrends
```

To analyze keyword trends on Google Search, use pytrends.

Here’s an example to track the trend of the keyword "data science" from 2019 to 2024:

```{code-cell} ipython3
from pytrends.request import TrendReq
```

```{code-cell} ipython3
pytrends = TrendReq(hl="en-US", tz=360)
pytrends.build_payload(kw_list=["data science"])

df = pytrends.interest_over_time()
df["data science"].plot(figsize=(20, 7))
```

[Link to pytrends](https://github.com/GeneralMills/pytrends)

+++

### snscrape: Scrape Social Networking Services in Python

+++

To scrape data from social media platforms like Twitter, Facebook, or Reddit, use `snscrape`.

Here's how to scrape all tweets from a user or get the latest 100 tweets with the hashtag #python.

+++

```bash
# Scrape all tweets from @KhuyenTran16
snscrape twitter-user KhuyenTran16

# Save outputs
snscrape twitter-user KhuyenTran16 >> khuyen_tweets 

# Scrape 100 tweets with hashtag python
snscrape --max-results 100 twitter-hashtag python
```

+++

[Link to snscrape](https://github.com/JustAnotherArchivist/snscrape).

+++

### Datacommons: Get Statistics about a Location in One Line of Code

```{code-cell} ipython3
:tags: [hide-cell]

!pip install datacommons
```

To get statistics about a location with one line of code, use `datacommons`.

```{code-cell} ipython3
import datacommons_pandas
import plotly.express as px 
import pandas as pd 
```

Here’s how to find median income in California over time:

```{code-cell} ipython3
median_income = datacommons_pandas.build_time_series("geoId/06", "Median_Income_Person")
median_income.index = pd.to_datetime(median_income.index)
median_income.plot(
    figsize=(20, 10),
    x="Income",
    y="Year",
    title="Median Income in California Over Time",
)
```

To visualize the number of people in the U.S. over time:

```{code-cell} ipython3
def process_ts(statistics: str):
    count_person = datacommons_pandas.build_time_series('country/USA', statistics)
    count_person.index = pd.to_datetime(count_person.index)
    count_person.name = statistics
    return count_person 
```

```{code-cell} ipython3
count_person_male = process_ts('Count_Person_Male')
count_person_female = process_ts('Count_Person_Female')
```

```{code-cell} ipython3
count_person = pd.concat([count_person_female, count_person_male], axis=1)

count_person.plot(
    figsize=(20, 10),
    title="Number of People in the U.S Over Time",
)
```

[Link to Datacommons](https://datacommons.org/).

+++

### Get Google News Using Python

```{code-cell} ipython3
:tags: [hide-cell]

!pip install GoogleNews
```

To retrieve Google News results for a specific keyword and date range, use the `GoogleNews` library.

Here's how you can get news articles related to a keyword within a given time period:

```{code-cell} ipython3
from GoogleNews import GoogleNews
googlenews = GoogleNews()
```

```{code-cell} ipython3
googlenews.set_time_range('02/01/2022','03/25/2022')
```

```{code-cell} ipython3
googlenews.search('funny')
```

```{code-cell} ipython3
googlenews.results()
```

[Link to GoogleNews](https://pypi.org/project/GoogleNews/).

+++

### people_also_ask: Python Wrapper for Google People Also Ask

```{code-cell} ipython3
:tags: [hide-cell]

!pip install people_also_ask 
```

To interact with Google's "People Also Ask" feature programmatically, use the `people_also_ask` library.

Here’s how to retrieve related questions and answers:

```{code-cell} ipython3
import people_also_ask as ask

ask.get_related_questions('data science')
```

```{code-cell} ipython3
ask.get_answer('Is data science a easy career?')
```

[Link to people-also-ask](https://github.com/lagranges/people_also_ask).

+++

### Scrape Facebook Public Pages Without an API Key

```{code-cell} ipython3
:tags: [hide-cell]

pip install facebook-scraper
```

To scrape Facebook public pages without needing an API key, use the `facebook-scraper` library.

You can use `facebook-scraper` to extract information from public profiles and groups. Here's how you can do it:

+++

Get group information:

```{code-cell} ipython3
from facebook_scraper import get_profile, get_group_info

get_group_info("thedachshundowners")
```

Get profile information

```{code-cell} ipython3
get_profile("zuck")
```

[Link to facebook-scraper](https://github.com/kevinzg/facebook-scraper).

+++

### Recipe-Scrapers: Automate Recipe Data Extraction

```{code-cell} ipython3
:tags: [hide-cell]

!pip install recipe-scrapers
```

For automated recipe data extraction, use the `recipe-scrapers` library. It simplifies the process of gathering recipe information from various websites.

```{code-cell} ipython3
from recipe_scrapers import scrape_me

scraper = scrape_me('https://cookieandkate.com/thai-red-curry-recipe/')

scraper.host()
```

```{code-cell} ipython3
scraper.title()
```

```{code-cell} ipython3
scraper.total_time()
```

```{code-cell} ipython3
scraper.ingredients()
```

```{code-cell} ipython3
scraper.instructions()
```

```{code-cell} ipython3
scraper.nutrients()
```

[Link to recipe-scrapers](https://bit.ly/3U6vw0w).

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Parsera: Natural Language Web Scraping with LLMs

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
!pip install parsera
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
!playwright install
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Writing and maintaining web scraping code requires constant updates due to changing HTML structures and complex selectors, which results in brittle code and frequent breakages.

With Parsera, you can scrape websites by simply describing what data you want to extract in plain language, letting LLMs handle the complexity of finding the right elements.

Here's an example that scrapes GitHub's trending Python repositories page to collect:
- Repository names
- Repository owners
- Star counts
- Fork counts

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import os

os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [remove-cell]
---
import nest_asyncio
nest_asyncio.apply()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
from parsera import Parsera
from pprint import pprint

url = "https://github.com/trending/python?since=daily"
elements = {
    "Repository": "Name of the repository",
    "Owner": "Owner of the repository",
    "Stars": "Number of stars",
    "Forks": "Number of forks",
}

scraper = Parsera()
result = scraper.run(url=url, elements=elements)
pprint(result)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

[Link to Parsera](https://github.com/raznem/parsera).

+++

### Generating Synthetic Tabular Data with TabGAN

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
!pip install tabgan
```

Limited, imbalanced, or missing data in tabular datasets can lead to poor model performance and biased predictions. 

To address this issue, TabGAN provides a solution by generating synthetic tabular data that maintains the statistical properties and relationships of the original dataset. 

In this example, we will demonstrate how to use TabGAN to generate high-quality synthetic data using different generators (GAN, Diffusion, or LLM-based).


First, we create random input data:

```{code-cell} ipython3
from tabgan.sampler import OriginalGenerator, GANGenerator, ForestDiffusionGenerator, LLMGenerator
import pandas as pd
import numpy as np


train = pd.DataFrame(np.random.randint(-10, 150, size=(150, 4)), columns=list("ABCD"))
target = pd.DataFrame(np.random.randint(0, 2, size=(150, 1)), columns=list("Y"))
test = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

print("Training Data:")
print(train.head())

print("\nTarget Data:")
print(target.head())

print("\nTest Data:")
print(test.head())
```

Next, we use the `OriginalGenerator` to generate synthetic data:

```{code-cell} ipython3
new_train1, new_target1 = OriginalGenerator().generate_data_pipe(train, target, test)

print("Training Data:")
print(new_train1.head())

print("\nTarget Data:")
print(new_target1.head())
```

The `generate_data_pipe` method takes in the following parameters:

*   `train_df`: The training dataframe.
*   `target`: The target variable for the training dataset.
*   `test_df`: The testing dataframe.
*   `deep_copy`: A boolean indicating whether to make a deep copy of the input dataframes. Default is `True`.
*   `only_adversarial`: A boolean indicating whether to only perform adversarial filtering on the training dataframe. Default is `False`.
*   `use_adversarial`: A boolean indicating whether to perform adversarial filtering on the generated data. Default is `True`.
*   `only_generated_data`: A boolean indicating whether to return only the generated data. Default is `False`.
    
Alternatively, we can use the `GANGenerator` to generate synthetic data:

```{code-cell} ipython3
new_train2, new_target2 = GANGenerator(
    gen_params={
        "batch_size": 500,  # Process data in batches of 500 samples at a time
        "epochs": 10,       # Train for a maximum of 10 epochs
        "patience": 5       # Stop early if there is no improvement for 5 epochs
}
).generate_data_pipe(train, target, test)

print("Training Data:")
print(new_train2.head())

print("\nTarget Data:")
print(new_target2.head())
```

The `GANGenerator` takes in the following parameters:

*   `gen_x_times`: A float indicating how much data to generate. Default is `1.1`.
*   `cat_cols`: A list of categorical columns. Default is `None`.
*   `bot_filter_quantile`: A float indicating the bottom quantile for post-processing filtering. Default is `0.001`.
*   `top_filter_quantile`: A float indicating the top quantile for post-processing filtering. Default is `0.999`.
*   `is_post_process`: A boolean indicating whether to perform post-processing filtering. Default is `True`.
*   `adversarial_model_params`: A dictionary of parameters for the adversarial filtering model. Default is `None`.
*   `pregeneration_frac`: A float indicating the fraction of data to generate before post-processing filtering. Default is `2`.
*   `gen_params`: A dictionary of parameters for the GAN training process. Default is `None`.

+++

[Link to TabGan](https://github.com/Diyago/Tabular-data-generation).
