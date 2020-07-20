# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <h1>NER using CRF<h1>
#
# <h2>Instructions<h2>
#
# - The header files and the data file(stored in path) is already loaded for you.
#
# - Start with exploring data and understanding different tags.
#
# - You must have noticed lot of NaN values, clean them up(Preferably using ffil)
#
# - Function Sentence() has been defined to group the dataframe into sentences. Pass your dataframe as a parameter to the function and store the output in sObject.
#
# - Take the sent parameter of sObject(i.e. sObject.sent) and store it in a variable called sentences. Print and see what it contains
#
# - Functions word2Features(To convert words into features), sent2features(To get features from sentences with the help of word2Features),sent2label(To get labels from sentences) are already defined for your help. Make sure you understand what these functions do and how they do it.
#
# - Get the features from sentences by passing the values of sentences(You will have to run a loop) to sent2features() function and store the features in variable X
#
# - Get the labels from sentences by passing the values of sentences(You will have to run a loop) to sent2labels() function and store the features in variable y
#
# - Split X,y into train and test for model fitting. Store them in X_train,X_test,y_train,y_test accordingly
#
# - Initialise a sklearn_crfsuite.CRF model called crf and fit X_train,y_train.
#
# - Use crf to predict from X_test and store the predicted values in y_pred
#
# - Use metrics.flat_classification_report to see the entire classification report between y_test and y_pred

# %%
# import required modules

import eli5
import joblib
import numpy as np
import pandas as pd
import sklearn_crfsuite
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics

# %%
path = "../input/ReutersNERDataset.csv"


# %%
# Read the file with the given encoding and do not throw any error, ignore it.
df = pd.read_csv(path, encoding="ISO-8859-1", error_bad_lines=False)


# %%
df.head()


# %%
df.isna().sum()


# %%
# filling NA values
df["Sentence #"].fillna(method="ffill", inplace=True)


# %%
df.head()


# %%
df.nunique()


# %%
df["sentenceNum"] = df["Sentence #"].str.split(":").apply(lambda x: x[1])


# %%
df["sentenceNum"] = df["sentenceNum"].astype("int")


# %%
df.dtypes


# %%
df.groupby("sentenceNum").agg({"Word": "count", "POS": "count", "Tag": "count"})


# %%
class Sentence(object):
    """Class for converting rows of words into sentence.
    Class has 3 attributes
    - data : stores the dataframe
    - grouped : tuple of word, pos and tag for each sentence in dataframe form
    - sent : list of list of tuple of word, pos and tag for each sentence

    Args:
        object ([pandas dataframe]): [dataframe having words, its postag and NER of sentence as rows]
    """

    data = None
    sent = None
    grouped = None

    def __init__(self, data):
        self.data = data
        # Take the data, extract out the word, part of speech associated and the Tag assigned and convert it
        # into a list of tuples.
        list_vals = lambda row: [
            (word, pos, tag)
            for word, pos, tag in list(zip(row["Word"], row["POS"], row["Tag"]))
        ]
        # Group the collected values according to the Sentence # column in the dataframe so that all the words
        # in a sentence are gouped together
        self.grouped = self.data.groupby("Sentence #").apply(list_vals)

        # Add the rows to the 'sent' list.
        self.sent = [row for row in self.grouped]


# %%
sObject = Sentence(df)


# %%
sentences = sObject.sent


# %%
print(sentences[:2])


# %%
def word2features(sent, i):
    """Function to create features that would be compatible with sklearn-crf package input definations.
    The inpuit to the api is a set of feature object which consists of the following features:
        - Whether or not the word is in lower case
        - The adjacent words to the word.
        - Where or not the word is in upper case.
        - Whether or not the word is a title or is a heading in the text.
        - If the word consist of digits only.
        - The POS tags of the word.
        - The POS tags of the adjacent words.

    Args:
        sent ([string]): [sentence for which feature needs to be created]
        i ([int]): [current row pointer]

    Returns:
        [dict]: [dictionary of features compatible with sklearn_crf API]
    """
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:postag": postag1,
                "-1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:postag": postag1,
                "+1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["EOS"] = True

    return features


# %%
def sent2features(sent):
    """function to get the feature dict for the sentences

    Args:
        sent ([string]): [sentence]

    Returns:
        [list]: [list of features for CRF]
    """
    return [word2features(sent, i) for i in range(len(sent))]


# %%
def sent2labels(sent):
    """Function to get labels for training CRF model

    Args:
        sent ([string]): [sentence]

    Returns:
        [list]: [list of NER tag for corresponding sentence]
    """
    return [label for token, postag, label in sent]


# %%
def sent2tokens(sent):
    return [token for token, postag, label in sent]


# %%
# Creating features for CRF model
X = [sent2features(sentence) for sentence in sentences]

# %%
# Create labels for CRF model
y = [sent2labels(sentence) for sentence in sentences]


# %%
# creating train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# %%
# fit a CRF model and predict labels
crf = sklearn_crfsuite.CRF(verbose=True)
crf.fit(X_train, y_train)
y_pred = crf.predict(X_test)


# %%
# print classification metrics
print(metrics.flat_classification_report(y_test, y_pred))
metrics.flat_f1_score(y_test, y_pred, average="weighted")

#%%
#
eli5.show_weights(crf, top=10)


# %%
# saving and loading a model
joblib.dump(crf, "../models/crf_model.joblib")
# joblib.load('../models/crf_model.joblib')


# %%
