# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import re
from re import error

import numpy as np
import pandas as pd
import seaborn as sns

# %%
from bs4 import BeautifulSoup

# %%
from IPython.core.interactiveshell import InteractiveShell
from matplotlib import pyplot as plt
from nltk.corpus import stopwords

# %%
InteractiveShell.ast_node_interactivity = "all"


# %%
data = pd.read_csv("../input/Reviews.csv.zip", keep_default_na=False)


# %%
data.head()
data.shape

# %%
contraction_mapping = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
}


# %%
stop_words = set(stopwords.words("english"))


# %%
def text_cleaner(text, num):
    # Remove all HTML tags, quotes, punctuations, apostrophes
    clean_txt = BeautifulSoup(text, "lxml").text
    clean_txt = re.sub(r"\([^)]*\)", "", clean_txt)
    clean_txt = re.sub('"', "", clean_txt)
    clean_txt = " ".join(
        [
            contraction_mapping[t] if t in contraction_mapping else t
            for t in clean_txt.split(" ")
        ]
    )
    clean_txt = re.sub(r"'s\b", "", clean_txt)
    clean_txt = re.sub("[^a-zA-Z]", " ", clean_txt)
    if num == 0:
        tokens = [w for w in clean_txt.split() if not w in stop_words]
    else:
        tokens = clean_txt.split()
    cleaned_text = []
    # remove words less than three characaters
    for token in tokens:
        if len(token) < 3:
            continue
        cleaned_text.append(token.lower())
    return (" ".join(cleaned_text)).strip()


# %%
# Preprocess reviews' text
cleaned_text = []
for t in data["Text"]:
    cleaned_text.append(text_cleaner(t, 0))


# %%
# preprocess summary
cleaned_summary = []
for t in data["Summary"]:
    cleaned_summary.append(text_cleaner(t, 1))

# %%
data["cleaned_text"] = cleaned_text
data["cleaned_summary"] = cleaned_summary


# %%
data.head()


# %%
data["cleaned_text_length"] = data["cleaned_text"].apply(len)
data["cleaned_summary_length"] = data["cleaned_summary"].apply(len)


# %%
sns.distplot(a=data["cleaned_text_length"])
#%%
sns.distplot(a=data["cleaned_summary_length"])

# %%
max_text_len = 314
max_summary_len = 27
#%%
cleaned_text = np.array(data["cleaned_text"])
cleaned_summary = np.array(data["cleaned_summary"])
#%%
short_text = []
short_summary = []
#%%
for i in range(len(cleaned_text)):
    if (
        len(cleaned_summary[i].split()) <= max_summary_len
        and len(cleaned_text[i].split()) <= max_text_len
    ):
        short_text.append(cleaned_text[i])
        short_summary.append(cleaned_summary[i])

df = pd.DataFrame({"text": short_text, "summary": short_summary})
#%%
df.head()


# %%
df["summary"] = df["summary"].apply(lambda x: "<START> " + x + " <END>")
#%%
df.to_csv("../tmp/train.csv", index=False)


# %%
