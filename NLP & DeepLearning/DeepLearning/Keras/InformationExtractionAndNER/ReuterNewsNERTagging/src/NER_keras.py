# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Import pandas library and fill the null values
#%%
import os

import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    TimeDistributed,
)

# Import Keras related modules and build the model
from keras.models import Input, Model, load_model

# We will be using keras built in function 'pad_sequences' to pad the input vectors to 'max_lan'
# This will ensure that all sequences in a list have the same length
from keras.preprocessing.sequence import pad_sequences

# Using the built-in function to_categorical to convert a class vector (integers) to binary class matrix.
from keras.utils import to_categorical

# Load the train_test_split model so that we can split the data into training data and test data
from sklearn.model_selection import train_test_split

#%%
# Set working as current directory
cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

#%%
# To view output of all cells
InteractiveShell.ast_node_interactivity = "all"


# %%
df = pd.read_csv(
    "../input/ReutersNERDataset.csv", encoding="ISO-8859-1", error_bad_lines=False
)
df = df.fillna(method="ffill")


# %%
# Same as last tutorial. To extract out values in Word, Tag and POS columns
class Sentence(object):
    data = None
    sent = None
    grouped = None

    def __init__(self, data):
        self.data = data
        list_vals = lambda row: [
            (word, pos, tag)
            for word, pos, tag in list(zip(row["Word"], row["POS"], row["Tag"]))
        ]
        self.grouped = self.data.groupby("Sentence #").apply(list_vals)
        self.sent = [row for row in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


# %%
# The input vector needs to be of equal & fixed length as defined by 'max_len'.
# We will use padding the sentences to 'max_len'
max_len = 50


# %%
# Get the words in form of a list and add the string "ENDPAD" at the end of the lsit
words = list(set(df["Word"].values))
n_words = len(words)
words.append("ENDPAD")


# %%
# Get all the tags as a list
tags = list(set(df["Tag"].values))


# %%
# As in the last turorial(Ner with CRF) we will reconstruct the input vectors
sObject = Sentence(df)
sentences = sObject.sent


# %%
sObject.get_next()


# %%
sObject.sent


# %%
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)


y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
y = [to_categorical(i, num_classes=len(tags)) for i in y]


# Split the data into Training data and Testing data. We keep 10% of the data/rows for testing our learned model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# %%
# We will now be building the model

# Input is used to instantiate a Keras tensor. A Keras tensor is a tensor object from the underlying backend(Tensorflow)
# which we augment with certain attributes that allow us to build a Keras model just buy knowing the inputs and
# output of the model
input = Input(shape=(max_len,))

# 'Embedding' turns positive integers into dense vectors of a fixed size
# Therefore, we supply to it the input/output dimesions, and the input length
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)

model = Dropout(0.1)(model)

model = BatchNormalization()(model)

# Initialize bi-directional LSTM cells
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(
    model
)

model = BatchNormalization()(model)

# Initialize a time distributed layer while building the sequential model
out = TimeDistributed(Dense(len(tags), activation="softmax"))(
    model
)  # softmax output layer
model = Model(input, out)


# %%
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#%%
trained = model.fit(
    X_train,
    np.array(y_train),
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    verbose=1,
)

#%%
# Loading keras model trained from Kaggle
model = load_model("../models/keras_model.h5")


# %%
p = model.predict(np.array([X_test[1234]]))
p = np.argmax(p, axis=1)


# %%
# Print the predictions of the sample # 1234
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(X_test[1234], p[0]):
    try:
        print("{:15}: {}".format(words[w], tags[pred]))
    except:
        pass
