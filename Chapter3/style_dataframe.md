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

## Style a DataFrame

+++

### Highlight Your pandas DataFrame for Easier Analysis

+++

Have you ever wanted to highlight your pandas DataFrame for easier analysis? For example, you might want positive values in green and negative ones in red.

That could be done with `df.style.apply`. 

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({"col1": [-5, -2, 1, 4], "col2": [2, 3, -1, 4]})
```

```{code-cell} ipython3
def highlight_number(row):
    return [
        "background-color: red; color: white"
        if cell <= 0
        else "background-color: green; color: white"
        for cell in row
    ]
```

```{code-cell} ipython3
df.style.apply(highlight_number)
```

One use case of highlighting is to compare the predictions of two or more models.

```{code-cell} ipython3
import pandas as pd

comparisons = pd.DataFrame(
    {
        "predictions_1": [1, 1, 1, 0, 1],
        "predictions_2": [0, 1, 0, 0, 0],
        "real_labels": [0, 1, 0, 0, 1],
    }
)


def highlight_cell(row):
    return [
        "background-color: red; color: white"
        if cell == 0
        else "background-color: green; color: white"
        for cell in row
    ]


comparisons.style.apply(highlight_cell)
```

### Color the Background of a pandas DataFrame in a Gradient Style

+++

If you want to color the background of a pandas DataFrame in a gradient style, use `df.style.background_gradient`. The color of the cell will change based on its value. 

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({"col1": [-5, -2, 1, 4], "col2": [2, 3, -1, 4]})
```

```{code-cell} ipython3
df.style.background_gradient()  
```

```{code-cell} ipython3
df.style.background_gradient(cmap='plasma')  
```

### Format the Text Display Value of Cells

+++

Sometimes, you might want to format your DataFrame before writing it to a file such as an Excel sheet. `df.style.format` allows you to do that.

```{code-cell} ipython3
import pandas as pd  
import numpy as np 

df = pd.DataFrame({'item': ['a', 'b'], 'price': [np.nan, 2.34]})
s = df.style.format({'item': str.title,'price': '${:.1f}'}, na_rep='MISSING')
s 
```

```{code-cell} ipython3
s.to_excel("formatted_file.xlsx")
```

### to_markdown: Print a DataFrame in Markdown Format

```{code-cell} ipython3
:tags: [hide-cell]

!pip install tabulate 
```

Sometimes, you might want to include a table in a markdown, such as GitHub README. If you want to print a DataFrame in markdown format, use `to_markdown()`.

```{code-cell} ipython3
import pandas as pd  

df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
print(df.to_markdown())
```

Copy and paste the output above in Jupyter Notebook's markdown cell will give you an output like below:

+++

|    |   a |   b |
|---:|----:|----:|
|  0 |   1 |   5 |
|  1 |   2 |   6 |
|  2 |   3 |   7 |
|  3 |   4 |   8 |

+++

You can also output markdown with a tabulate option:

```{code-cell} ipython3
print(df.to_markdown(tablefmt="grid"))
```

### Print a DataFrame in Latex Format

+++

If you want to print a DataFrame in LaTeX format, use `df.style.to_latex()`. This is useful when you want to include your DataFrame in a LaTeX editor.

```{code-cell} ipython3
import pandas as pd 

df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [5, 6, 7, 8]})
print(df.style.to_latex())
```

![](../img/equation4.png)

+++

You can also specify the style of the table before turning it to LaTeX

```{code-cell} ipython3
latex = df.style.set_table_styles(
    [
        {"selector": "toprule", "props": ":hline;"},
        {"selector": "midrule", "props": ":hline;"},
        {"selector": "bottomrule", "props": ":hline;"},
    ]
).to_latex(column_format="|l|l|l|")
print(latex)
```

![](../img/equation5.png)
