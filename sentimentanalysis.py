#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings("ignore")

# Download NLTK requirements (only first time)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# ------------------------------------
# 1. Load tweets dataset
# ------------------------------------
# Example dataset path — change based on your file
df = pd.read_csv("tweets.csv")   # Must contain a column named "text"

print("Sample tweets:")
print(df.head())

# ------------------------------------
# 2. Cleaning Functions
# ------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()                                        # Lowercase
    text = re.sub(r"http\S+|www.\S+", "", text)                # Remove URLs
    text = re.sub(r"@\w+", "", text)                           # Remove @mentions
    text = re.sub(r"#\w+", "", text)                           # Remove hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)                    # Remove special chars
    tokens = nltk.word_tokenize(text)                          # Tokenize
    tokens = [w for w in tokens if w not in stop_words]        # Remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens]         # Lemmatization
    return " ".join(tokens)

df["cleaned_text"] = df["text"].apply(clean_text)

print("\nCleaned Tweets:")
print(df.head())

# ------------------------------------
# 3. Sentiment Analysis using TextBlob
# ------------------------------------
def get_sentiment(text):
    score = TextBlob(text).sentiment.polarity

    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["cleaned_text"].apply(get_sentiment)

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# ------------------------------------
# 4. Visualization of Sentiments
# ------------------------------------
plt.figure(figsize=(6,6))
df["Sentiment"].value_counts().plot(kind="pie", autopct='%1.1f%%')
plt.title("Tweet Sentiment Distribution")
plt.ylabel("")
plt.show()

plt.figure(figsize=(7,5))
df["Sentiment"].value_counts().plot(kind="bar")
plt.title("Sentiment Counts")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.show()

# ------------------------------------
# 5. Save final results
# ------------------------------------
df.to_csv("tweet_sentiment_output.csv", index=False)
print("\nFinal dataset saved as tweet_sentiment_output.csv")


# In[13]:


pip install textblob


# In[ ]:




