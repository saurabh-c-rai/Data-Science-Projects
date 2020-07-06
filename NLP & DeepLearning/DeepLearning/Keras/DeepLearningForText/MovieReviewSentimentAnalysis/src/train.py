#%%

import ast
import os

import keras
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Sequential, optimizers, regularizers
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
)
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)


# %%
train_reviews = pd.read_csv("../input/train.csv")
train_reviews.head()

# %%
test_reviews = pd.read_csv("../input/test.csv")
test_reviews.head()

#%%
train_reviews["reviews"] = train_reviews["reviews"].apply(ast.literal_eval)

test_reviews["reviews"] = test_reviews["reviews"].apply(ast.literal_eval)
#%%
train_reviews["review_len"] = train_reviews["reviews"].apply(len)
test_reviews["review_len"] = test_reviews["reviews"].apply(len)
#%%
sns.distplot(a=train_reviews["review_len"])

#%%
sns.distplot(a=test_reviews["review_len"])

#%%
train_padded_sequence = pad_sequences(train_reviews["reviews"], maxlen=1500)
test_padded_sequence = pad_sequences(test_reviews["reviews"], maxlen=1500)

#%%
# vocab size
train_reviews["max_index"] = train_reviews["reviews"].apply(max)
#%%
# Defining shape of input and output & Reguralization Parameter
INPUT_SHAPE = train_padded_sequence.shape[1]
vocab_size = max(train_reviews["max_index"])
DROPOUT_RATE = 0.3
L1_PENALTY = 0.0001
L2_PENALTY = 0.0001

#%%
# Instantiating sequential model
model = Sequential()
# model.add(Dense(input_shape=(INPUT_SHAPE,), units=max_len))
model.add(
    Embedding(
        input_dim=vocab_size, output_dim=300, input_length=INPUT_SHAPE, trainable=True,
    )
)
model.add(BatchNormalization())

# Adding hidden layers
model.add(
    Dense(
        units=8,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=L1_PENALTY, l2=L2_PENALTY),
    )
)
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

model.add(LSTM(64))
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

model.add(
    Dense(
        units=8,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=L1_PENALTY, l2=L2_PENALTY),
    )
)
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

model.add(Flatten())

# Adding output layer
model.add(Dense(1, activation="sigmoid"))

# %%
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# %%
model.compile(optimizer=adam, metrics=["accuracy"], loss="binary_crossentropy")
model.summary()
# %%
model.fit(
    x=train_padded_sequence,
    y=train_reviews["sentiments"].values,
    epochs=100,
    verbose=1,
    batch_size=32,
    validation_split=0.2,
)


# %%
y_pred = model.predict(test_padded_sequence)

test_reviews["sentiments"] = (y_pred >= 0.5) * 1

test_reviews.loc[:, ["id", "sentiments"]].to_csv("submission.csv", index=False)
# %%
