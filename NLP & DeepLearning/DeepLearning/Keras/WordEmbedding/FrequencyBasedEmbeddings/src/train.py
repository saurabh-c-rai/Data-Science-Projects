#%%
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#%%

corpus = [
    "Alice was beginning to get very tired of sitting by her sister on the bank",
    "What is the use of a book, thought Alice `without pictures or conversation",
    "There was nothing so very remarkable in that",
    "The Rabbit actually took a watch out its waist",
    "Alice started to her feet",
    "Alice opened the door and found that it led into a small passage",
    "And she went on planning to herself how she would manage it",
    "Alice took up the fan and gloves",
]

#%%

stop_words = nltk.corpus.stopwords.words("english")

#%%
def preprocess_document(corpus):

    # lower the string and strip spaces
    corpus = corpus.lower()
    corpus = corpus.strip()

    # tokenize the words in document
    word_tokens = nltk.WordPunctTokenizer().tokenize(corpus)

    # remove stopwords
    filtered_tokens = [token for token in word_tokens if token not in stop_words]

    # join document from the tokens
    corpus = " ".join(filtered_tokens)

    return corpus


#%%
# Loading the data
corpus_df = pd.DataFrame({"Sentences": corpus})

#%%
# Vectorizing function so that it can work on corpus
preprocess_document = np.vectorize(preprocess_document)

# %%
clean_corpus = preprocess_document(corpus_df)


# %%
countVector = CountVectorizer()

# %%
cv_matrix = countVector.fit_transform(clean_corpus.ravel()).toarray()
# %%
cv_vocab = countVector.get_feature_names()

# %%
cv_df = pd.DataFrame(data=cv_matrix, columns=cv_vocab)
cv_df.head()
# %%
