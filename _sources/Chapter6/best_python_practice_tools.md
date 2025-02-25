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

## Tools for Best Python Practices

+++

This section cover tools that encourage best Python practices.

+++

### Managing Default Configurations with Hydra

```{code-cell} ipython3
:tags: [hide-cell]

!pip install hydra-core --upgrade
```

Managing default configurations across different environments or experiments is cumbersome, often requiring explicit specification of config options every time you run your application. Data scientists frequently need to switch between different database connections or model parameters, leading to repetitive command-line arguments.

+++

Hydra solves this by allowing you to set and override default configurations easily. Here's how to use it:

First, create the configuration files:

```{code-cell} ipython3
%mkdir conf
%mkdir conf/db
```

```{code-cell} ipython3
%%writefile conf/config.yaml
defaults:
  - db: mysql  # Set mysql as default database
```

```{code-cell} ipython3
%%writefile conf/db/mysql.yaml
driver: mysql
user: omry
pass: secret
```

```{code-cell} ipython3
%%writefile conf/db/postgresql.yaml
driver: postgresql
user: postgres_user
pass: drowssap
```

Create the Python application:

```{code-cell} ipython3
%%writefile main.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()
```

You can run with default config (mysql) or override to use postgresql:

```{code-cell} ipython3
!python main.py # Uses mysql by default
```

```{code-cell} ipython3
!python main.py db=postgresql  # Override to use postgresql
```

[Link to Hydra](https://github.com/facebookresearch/hydra)

+++

### Store Sensitive Information Securely in Python with .env Files

```{code-cell} ipython3
:tags: [hide-cell]

!pip install python-dotenv
```

Managing configuration and sensitive data in code results in security risks and deployment challenges as values are hard-coded or need to be manually set in different environments. This causes maintenance overhead and potential security breaches.

```{code-cell} ipython3
PASSWORD=123
USERNAME=myusername
```

Python-dotenv lets you separate configuration from code by loading environment variables from a `.env` file. You can:

- Keep sensitive data out of code
- Use different configurations per environment

Here is an example:

```{code-cell} ipython3
%%writefile .env
PASSWORD=123
USERNAME=myusername
```

```{code-cell} ipython3
from dotenv import load_dotenv
import os 

load_dotenv()
PASSWORD = os.getenv('PASSWORD')
USERNAME = os.getenv('USERNAME')
print(PASSWORD)
print(USERNAME)
```

[Link to python-dotenv](https://github.com/theskumar/python-dotenv)

+++

### docopt: Create Beautiful Command-line Interfaces for Documentation in Python

```{code-cell} ipython3
:tags: [hide-cell]

!pip install docopt 
```

Writing documentation for your Python script helps others understand how to use your script. However, instead of making them spend some time to find the documentation in your script, wouldn’t it be nice if they can view the documentation in the terminal?

That is when docopt comes in handy. docopt allows you to create beautiful command-line interfaces by passing a Python string.

+++

To understand how docopt works, we can add a docstring at the beginning of the file named `docopt_example.py`.

```{code-cell} ipython3
%%writefile docopt_example.py
"""Extract keywords of an input file
Usage:
    docopt_example.py --data-dir=<data-directory> [--input-path=<path>]
Options:
    --data-dir=<path>    Directory of the data
    --input-path=<path>  Name of the input file [default: input_text.txt]
"""

from docopt import docopt 

if __name__ == '__main__':
    args = docopt(__doc__, argv=None, help=True)
    data_dir = args['--data-dir']
    input_path = args['--input-path']

    if data_dir:
        print(f"Extracting keywords from {data_dir}/{input_path}")
```

Running the file `docopt_example.py` should give us the output like below:

```bash
$ python docopt_example.py
```

```{code-cell} ipython3
:tags: [hide-input]

!python docopt_example.py
```

[Link to docopt](http://docopt.org/).
