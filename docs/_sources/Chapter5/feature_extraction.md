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

```{code-cell} ipython3
:tags: [remove-cell]

import warnings
warnings.filterwarnings("ignore")
```

## Feature Extraction

+++

### distfit: Find The Best Theoretical Distribution For Your Data

```{code-cell} ipython3
:tags: [hide-cell]

!pip install distfit
```

If you're looking to identify the best theoretical distribution for your data in Python, try distfit. It allows you to fit and compare multiple distributions, identifying the best match for your dataset.

```{code-cell} ipython3
import numpy as np
from distfit import distfit

X = np.random.normal(0, 3, 1000)

# Initialize model
dist = distfit()

# Find best theoretical distribution for empirical data X
distribution = dist.fit_transform(X)
dist.plot()
```

Beyond finding the optimal distribution, `distfit` can also help identify outliers based on deviation from the fitted distribution.

+++

[Link to distfit](https://erdogant.github.io/distfit/pages/html/index.html).

+++

### Geopy: Extract Location Based on Python String

```{code-cell} ipython3
:tags: [hide-cell]

!pip install geopy
```

`Geopy` simplifies the process of extracting geospatial information from location strings. With just a few lines of code, you can obtain the coordinates of addresses globally.

```{code-cell} ipython3
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="find_location")
location = geolocator.geocode("30 North Circle Drive")
```

To get detailed information about the location:

```{code-cell} ipython3
location.address
```

You can also extract latitude and longitude:

```{code-cell} ipython3
location.latitude, location.longitude
```

[Link to Geopy](https://geopy.readthedocs.io/en/stable/)

+++

### fastai’s cont_cat_split: Separate Continuous and Categorical Variables

```{code-cell} ipython3
:tags: [hide-cell]

!pip install fastai
```

Fastai's `cont_cat_split` method helps you automatically separate continuous and categorical columns in a DataFrame based on their cardinality.

```{code-cell} ipython3
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
```

```{code-cell} ipython3
cont_names, cat_names = cont_cat_split(df, max_card=3)
print("Continuous columns:", cont_names)
print("Categorical columns:", cat_names)
```

[Link to the documentation](https://docs.fast.ai/tabular.core.html).

+++

### Patsy: Build Features with Arbitrary Python Code

```{code-cell} ipython3
:tags: [hide-cell]

!pip install patsy
```

Patsy lets you quickly create features for your model using an intuitive syntax, ideal for experimentation.

```{code-cell} ipython3
from sklearn.datasets import load_wine
import pandas as pd 
```

```{code-cell} ipython3
df = load_wine(as_frame=True)
data = pd.concat([df['data'], df['target']], axis=1)
data.head(10)
```

```{code-cell} ipython3
from patsy import dmatrices

y, X = dmatrices('target ~ alcohol + flavanoids + proline', data=data)
```

```{code-cell} ipython3
X
```

These features can be directly used with machine learning models:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, y)
```

[Link to Patsy](https://patsy.readthedocs.io/en/latest/overview.html).

+++

### yarl: Create and Extract Elements from a URL Using Python

```{code-cell} ipython3
:tags: [hide-cell]

!pip install yarl
```

`yarl` makes URL parsing and creation easy. You can extract elements like host, path, and query from a URL or construct new URLs.

```{code-cell} ipython3
from yarl import URL 

url = URL('https://github.com/search?q=data+science')
url 
```

```{code-cell} ipython3
print(url.host) 
```

```{code-cell} ipython3
print(url.path) 
```

```{code-cell} ipython3
print(url.query_string) 
```

You can also build new URLs:

```{code-cell} ipython3
# Create a URL

url = URL.build(
    scheme="https",
    host="github.com",
    path="/search",
    query={"p": 2, "q": "data science"},
)

print(url)
```

```{code-cell} ipython3
# Replace the query

print(url.with_query({"q": "python"}))
```

```{code-cell} ipython3
# Replace the path

new_path = url.with_path("khuyentran1401/Data-science")
print(new_path)
```

```{code-cell} ipython3
# Update the fragment

print(new_path.with_fragment("contents"))
```

[Link to yarl](https://github.com/aio-libs/yarl).

+++

### Pigeon: Quickly Annotate Your Data on Jupyter Notebook

```{code-cell} ipython3
:tags: [hide-cell]

!pip install pigeon-jupyter
```

For fast data annotation within Jupyter Notebooks, use `Pigeon`. This tool allows you to label data interactively by selecting from predefined options.

```{code-cell} ipython3
from pigeon import annotate


annotations = annotate(
    ["The service is terrible", "I will definitely come here again"],
    options=["positive", "negative"],
)
```

```{code-cell} ipython3
annotations
```

![](../img/pigeon_demo.gif)

+++

After labeling all your data, you can get the examples along with their labels by calling `annotations`.

+++

[Link to Pigeon](https://github.com/agermanidis/pigeon)

+++

### probablepeople: Parse Unstructured Names Into Structured Components

```{code-cell} ipython3
:tags: [hide-cell]

!pip install probablepeople  
```

`probablepeople` helps you parse unstructured names into structured components like first names, surnames, and company names.

```{code-cell} ipython3
import probablepeople as pp

pp.parse("Mr. Owen Harris II")
```

```{code-cell} ipython3
pp.parse("Kate & John Cumings")
```

```{code-cell} ipython3
pp.parse("Prefect Technologies, Inc")
```

[Link to probablepeople](https://github.com/datamade/probablepeople).

+++

### Supercharge PDF Text Extraction in Python with pypdf

```{code-cell} ipython3
:tags: [hide-cell]

!pip install -U pypdf
```

PDF text is designed for beautiful on-screen display rather than optimized structured data extraction, making text extraction from PDFs challenging. 

Besides simple text extraction, pypdf also knows about fonts, encodings, and typical character distance, which enhances the accuracy of text extraction from PDFs.

```{code-cell} ipython3
from pypdf import PdfReader

reader = PdfReader("example.pdf")
page = reader.pages[0]
text = page.extract_text()
```

[Link to pypdf](https://github.com/py-pdf/pypdf).
