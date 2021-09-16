import pytest 
from textblob import TextBlob

def extract_sentiment(text: str):
    """Extract sentimetn using textblob. Polarity is within range [-1, 1]"""
    
    text = TextBlob(text)
    return text.sentiment.polarity

@pytest.fixture 
def example_data():
    return 'Today I found a duck and I am happy'

def test_extract_sentiment(example_data):
    sentiment = extract_sentiment(example_data)
    assert sentiment > 0