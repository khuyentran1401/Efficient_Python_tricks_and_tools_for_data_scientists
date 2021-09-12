#!/usr/bin/env python
# coding: utf-8

# ## Natural Language Processing

# This section some tools to process and work with text.

# ### TextBlob: Processing Text in One Line of Code

# Processing text doesn’t need to be hard. If you want to find the sentiment of the text, tokenize text, find noun phrase and word frequencies, correct spelling, etc in one line of code, try TextBlob.
# 

# In[1]:


get_ipython().system('python -m textblob.download_corpora')


# In[2]:


from textblob import TextBlob

text = "Today is a beautiful day"
blob = TextBlob(text)

blob.words # Word tokenization


# In[3]:


blob.noun_phrases # Noun phrase extraction


# In[4]:


blob.sentiment # Sentiment analysis


# In[5]:


blob.word_counts # Word counts


# In[6]:


# Spelling correction
text = "Today is a beutiful day"
blob = TextBlob(text)
blob.correct()


# [Link to TextBlob](https://textblob.readthedocs.io/en/dev/).
# 
# [Link to my article about TextBlob](https://towardsdatascience.com/supercharge-your-python-string-with-textblob-2d9c08a8da05?sk=b9de5981cf74c0adf8d9f2a913e3ca05).

# ### sumy: Summarize Text in One Line of Code

# If you want to summarize text using Python or command line, try sumy.
# 
# The great things about sumy compared to other summarization tools are that it is easy to use and it allows you to use 7 different methods to summarize the text.
# 
# Below is how sumy summarizes the article How to Learn Data Science (Step-By-Step) in 2020 at DataQuest.

# ```bash
# $ sumy lex-rank --length=10 --url=https://www.dataquest.io/blog/learn-data-science/ 
# ```

# In[7]:


get_ipython().system('sumy lex-rank --length=10 --url=https://www.dataquest.io/blog/learn-data-science/ ')


# [Link to Sumy](https://github.com/miso-belica/sumy).

# ### Spacy_streamlit: Create a Web App to Visualize Your Text in 3 Lines of Code

# If you want to quickly create an app to visualize the structure of a text, try spacy_streamlit. 

# To understand how to use spacy_streamlit, we add the code below to a file called `streamlit_app.py`:
# ```python
# # streamlit_app.py
# import spacy_streamlit 
# 
# models = ['en_core_web_sm']
# text = "Today is a beautiful day"
# spacy_streamlit.visualize(models, text)
# ```

# On your terminal, type:
# ```bash
# $ streamlit run streamlit_app.py
# ```
# Output:

# In[8]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[9]:


get_ipython().system('streamlit run streamlit_app.py')


# Click the URL and you should see something like below:

# ![image](streamlit_app.png)

# ### textacy: Extract a Contiguous Sequence of 2 Words

# If you want to extract a contiguous sequence of 2 words, for example, 'data science', not 'data', what should you do? That is when the concept of extracting n-gram from text becomes useful.
# 
# A really useful tool to easily extract n-gram with a specified number of words in the sequence is `textacy`. 

# In[17]:


import pandas as pd 
import spacy 
from textacy.extract import ngrams

nlp = spacy.load('en_core_web_sm')

text = nlp('Data science is an inter-disciplinary field that uses'
' scientific methods, processes, algorithms, and systme to extract'
' knowledge and insights from many structural and unstructured data.')

n_grams = 2 # contiguous sequence of a word
min_freq = 1 # extract n -gram based on its frequency

pd.Series([n.text for n in ngrams(text, n=n_grams, min_freq=1)]).value_counts()


# [Link to textacy](https://textacy.readthedocs.io/en/stable/quickstart.html#working-with-text)

# ### difflib: Detect The “Almost Similar” Articles

# When analyzing articles, different articles can be almost similar but not 100% identical, maybe because of the grammar, or because of the change in two or three words (such as cross-posting). How can we detect the “almost similar” articles and drop one of them? That is when `difflib.SequenceMatcher` comes in handy. 

# In[19]:


from difflib import SequenceMatcher

text1 = 'I am Khuyen'
text2 = 'I am Khuen'
print(SequenceMatcher(a=text1, b=text2).ratio())


# ### Convert Number to Words
# 

# If there are both number 105 and the words ‘one hundred and five’ in a text, they should deliver the same meaning. How can we map 105 to ‘one hundred and five’? There is a Python libary to convert number to words called `num2words`.

# In[21]:


from num2words import num2words

num2words(105)


# In[22]:


num2words(105, to='ordinal')


# The library can also generate ordinal numbers and support multiple languages! 

# In[23]:


num2words(105, lang='vi')


# In[25]:


num2words(105, lang='es')


# [Link to num2words](https://github.com/savoirfairelinux/num2words).

# ### texthero.clean: Preprocess Text in One Line of Code

# If you want to preprocess text in one line of code, try texthero. The `texthero.clean` method will:
# 
# - fill missing values
# - convert upper case to lower case
# - remove digits
# - remove punctuation
# - remove stopwords
# - remove whitespace
# 
# The code below shows an example of `texthero.clean`.

# In[27]:


import numpy as np
import pandas as pd
import texthero as hero

df = pd.DataFrame(
    {
        "text": [
            "Today is a    beautiful day",
            "There are 3 ducks in this pond",
            "This is. very cool.",
            np.nan,
        ]
    }
)

df.text.pipe(hero.clean)


# Texthero also provides other useful methods to process and visualize text.
# 
# [Link to texthero](https://github.com/jbesomi/texthero).

# ### wordfreq: Estimate the Frequency of a Word in 36 Languages

# If you want to look up the frequency of a certain word in your language, try wordfreq.
# 
# wordfreq supports 36 languages. wordfreq even covers words that appear at least once per 10 million words.

# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordfreq import word_frequency

word_frequency("eat", "en")


# In[32]:


word_frequency("the", "en")


# In[33]:


sentence = "There is a dog running in a park"
words = sentence.split(" ")
word_frequencies = [word_frequency(word, "en") for word in words]

sns.barplot(words, word_frequencies)
plt.show()


# [Link to wordfreq](https://github.com/khuyentran1401/Python-data-science-code-snippet/blob/master/code_snippets/data_science_tools/wordfreq_example.py).

# ### newspaper3k: Extract Meaningful Information From an Articles in 2 Lines of Code

# If you want to quickly extract meaningful information from an article in a few lines of code, try newspaper3k. 

# In[20]:


from newspaper import Article
import nltk

nltk.download("punkt")


# In[21]:


url = "https://www.dataquest.io/blog/learn-data-science/"
article = Article(url)
article.download()
article.parse()


# In[22]:


article.title


# In[23]:


article.publish_date


# In[24]:


article.top_image


# In[25]:


article.nlp()


# In[26]:


article.summary


# In[27]:


article.keywords


# [Link to newspaper3k](https://newspaper.readthedocs.io/en/latest/user_guide/quickstart.html?highlight=parse#performing-nlp-on-an-article).
# 
# 
