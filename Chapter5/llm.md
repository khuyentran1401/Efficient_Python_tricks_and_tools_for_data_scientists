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

## Large Language Model (LLM)

+++

### Simplify LLM Integration with Magentic's @prompt Decorator

```{code-cell} ipython3
:tags: [hide-cell]

!pip install magentic
```

To enhance your code's natural language skills with LLM effortlessly, try magentic. 

With magentic, you can use the `@prompt` decorator to create functions that return organized LLM results, keeping your code neat and easy to read.

```{code-cell} ipython3
import openai

openai.api_key = "sk-..."
```

```{code-cell} ipython3
from magentic import prompt


@prompt('Add more "dude"ness to: {phrase}')
def dudeify(phrase: str) -> str:
    ...  # No function body as this is never executed


dudeify("Hello, how are you?")
# "Hey, dude! What's up? How's it going, my man?"
```

The `@prompt` decorator will consider the return type annotation, including those supported by pydantic.

```{code-cell} ipython3
from magentic import prompt, FunctionCall
from pydantic import BaseModel
from typing import Literal


class MilkTea(BaseModel):
    tea: str
    sweetness_percentage: float
    topping: str


@prompt("Create a milk tea with the following tea {tea}.")
def create_milk_tea(tea: str) -> MilkTea:
    ...


create_milk_tea("green tea")
```

The `@prompt` decorator also considers a function call.

```{code-cell} ipython3
def froth_milk(temperature: int, texture: Literal["foamy", "hot", "cold"]) -> str:
    """Froth the milk to the desired temperature and texture."""
    return f"Frothing milk to {temperature} F with texture {texture}"


@prompt(
    "Prepare the milk for my {coffee_type}",
    functions=[froth_milk],
)
def configure_coffee(coffee_type: str) -> FunctionCall[str]:
    ...


output = configure_coffee("latte!")
output()
```

[Link to magentic](https://github.com/jackmpcollins/magentic).

+++

### Outlines: Ensuring Consistent Outputs from Language Models

+++

The Outlines library enables controlling the outputs of language models. This makes the outputs more predictable, ensuring the reliability of systems using large language models.

+++

```python
import outlines

model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""
# Only return a choice between multiple possibilities
answer = outlines.generate.choice(model, ["Positive", "Negative"])(prompt)
```

+++

```python
# Only return integers or floats
model = outlines.models.transformers("mistralai/Mistral-7B-v0.1")

prompt = "1+1="
answer = outlines.generate.format(model, int)(prompt)

prompt = "sqrt(2)="
answer = outlines.generate.format(model, float)(prompt)
```

+++

[Link to Outlines](https://github.com/outlines-dev/outlines).

+++

### Mirascope: Extract Structured Data Extraction From LLM Outputs

```{code-cell} ipython3
:tags: [hide-cell]

!pip install mirascope
```

Large Language Models (LLMs) are powerful at producing human-like text, but their outputs lack structure, which can limit their usefulness in many practical applications that require organized data.

Mirascope offers a solution by enabling the extraction of structured information from LLM outputs reliably.

The following code uses Mirascope to extract meeting details such as topic, date, time, and participants.

```{code-cell} ipython3
import os


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
```

```{code-cell} ipython3
from typing import List, Type
from pydantic import BaseModel
from mirascope.openai import OpenAIExtractor


class MeetingDetails(BaseModel):
    topic: str
    date: str
    time: str
    participants: List[str]


class MeetingExtractor(OpenAIExtractor[MeetingDetails]):
    extract_schema: Type[MeetingDetails] = MeetingDetails
    prompt_template = """
    Extract the meeting details from the following description:
    {description}
    """

    description: str


# Example usage:
description = "Discuss the upcoming product launch on June 15th at 3 PM with John, Sarah, and Mike."
meeting_details = MeetingExtractor(description=description).extract()
assert isinstance(meeting_details, MeetingDetails)
print(meeting_details)
```

[Link to Mirascope](https://bit.ly/4bkciv3).

+++

### Maximize Accuracy and Relevance with External Data and LLMs

```{code-cell} ipython3
:tags: [hide-cell]

!pip install -U mirascope
```

Combining external data and an LLM offers the best of both worlds: accuracy and relevance. External data provides up-to-date information, while an LLM can generate text based on input prompts. Together, they enable a system to respond helpfully to a wider range of queries.

Mirascope simplifies this combination with Pythonic code. In the example below, we use an LLM to process natural language prompts and query the database for data.

```{code-cell} ipython3
:tags: [hide-cell]

import sqlite3

# Set up database and table for the example below
conn = sqlite3.connect("grocery.db")
cursor = conn.cursor()

# Create the 'grocery_items' table
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS grocery_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL
    )
"""
)

# Insert some sample data
items = [
    ("apple", "Fruits", 0.75),
    ("banana", "Fruits", 0.50),
    ("carrot", "Vegetables", 1.20),
]

cursor.executemany(
    "INSERT INTO grocery_items (name, category, price) VALUES (?, ?, ?)", items
)

# Commit the changes and close the connection
conn.commit()
conn.close()
print("Database created with sample data.")
```

```{code-cell} ipython3
import os


os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
```

```{code-cell} ipython3
from mirascope.openai import OpenAICall, OpenAICallParams
import sqlite3

# Assume you have a SQLite database with a 'grocery_items' table
conn = sqlite3.connect("grocery.db")


def get_item_info(table: str, item: str, info: str) -> dict:
    """Get `info` from the `table` table based on `item`."""
    cursor = conn.cursor()
    try:
        row = cursor.execute(
            f"SELECT {info} FROM {table} WHERE name = ?", (item,)
        ).fetchone()
        return f"The {info} for {item} is {row[0]}."
    except TypeError:
        return f"Sorry but {item} doesn't exist in the database."


class GroceryItemQuery(OpenAICall):
    prompt_template = """
    SYSTEM:
    Your task is to query a database given a user's input.

    USER:
    {input}
    """
    input: str
    call_params = OpenAICallParams(tools=[get_item_info])


text = "What's the price for banana in the grocery_items table?"
query_tool = GroceryItemQuery(input=text).call().tool
result = query_tool.fn(**query_tool.args)
result
```

[Link to Mirascope](https://bit.ly/4awfNhg).

```{code-cell} ipython3
!pip install chromadb 'numpy<2'
```

Managing and querying large collections of text data using traditional databases or simple search methods results in poor semantic matches and complex implementation. This causes difficulties in building AI applications that need to find contextually similar content.

```{code-cell} ipython3
# Traditional approach with basic text search
documents = [
    "The weather is great today",
    "The climate is excellent",
    "Machine learning models are fascinating",
]

# Search by exact match or simple substring
query = "How's the weather?"
results = [doc for doc in documents if "weather" in doc.lower()]

# Only finds documents with exact word "weather", misses semantically similar ones
print(results)
```

You can use Chroma to easily store and query documents using their semantic meaning through embeddings. The tool handles the embedding creation and similarity search automatically, making it simple to build AI applications with semantic search capabilities.


```{code-cell} ipython3
import chromadb

# Initialize client and collection
client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents
collection.add(
    documents=[
        "The weather is great today",
        "The climate is excellent",
        "Machine learning models are fascinating"
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Query semantically similar documents
results = collection.query(
    query_texts=["How's the weather?"],
    n_results=2
)
# Returns both weather and climate documents due to semantic similarity
print(results['documents'])
```

The example shows how Chroma automatically converts text into embeddings and finds semantically similar documents, even when they don't share exact words. This makes it much easier to build applications that can understand the meaning of text, not just match keywords.
