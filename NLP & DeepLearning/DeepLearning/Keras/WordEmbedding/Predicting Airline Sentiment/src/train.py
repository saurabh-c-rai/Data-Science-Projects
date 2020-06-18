#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

#%%
# Import libraries
from string import punctuation

# %%
import keras
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from keras import Sequential, optimizers, regularizers
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
)
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# %%
from keras.utils import to_categorical
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer, WordPunctTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# %%
# Read train & test data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
master_df = pd.read_csv("../tmp/17_742210_compressed_Tweets.csv.zip", compression="zip")


#%%
print(f"Train data has {train_df.shape[0]} rows and {train_df.shape[1]} columns")
print(f"Test data has {test_df.shape[0]} rows and {test_df.shape[1]} columns")
print(f"Test data has {master_df.shape[0]} rows and {master_df.shape[1]} columns")
# %%
# Proportion of target class in train data
train_df["airline_sentiment"].value_counts() / train_df[
    "airline_sentiment"
].value_counts().sum()

# %%
print("Target encoding")
print("0 : Bad Review")
print("1 : Neutral Review")
print("2 : Good Review")
target_encoding = {"negative": 0, "positive": 2, "neutral": 1}

# %%
# Initializing NLTK classes
stopword = stopwords.words("english")
mystopword = list(stopword) + list(punctuation)
tokenizer = TweetTokenizer(strip_handles=True,)
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

# %%
# Defining function for preprocessing
def preprocess_document(corpus):
    """Function to preprocess the corpus. Following actions will be performed :-
    - words will be converted to lower case
    - redundant spaces will be removed
    - stopwords from nltk library will be removed 

    Arguments:
        corpus {[String]} -- [Sentences]

    Returns:
        [String] -- [Cleaned sentence]
    """
    # lower the string and strip spaces
    corpus = corpus.lower()
    corpus = corpus.strip()

    # tokenize the words in document
    word_tokens = tokenizer.tokenize(corpus)

    # remove stopwords
    # filtered_tokens = [token for token in word_tokens if token not in mystopword]

    # lemmatized
    lemmatized_token = [lemma.lemmatize(token) for token in word_tokens]

    # join document from the tokens
    corpus = " ".join(lemmatized_token)

    return corpus


#%%time
train_df["cleaned_tweets"] = train_df["text"].apply(preprocess_document)
test_df["cleaned_tweets"] = test_df["text"].apply(preprocess_document)

#%%
data = train_df[["cleaned_tweets", "airline_sentiment"]]
test_data = test_df[["tweet_id", "cleaned_tweets"]]


# %%
data.head()


# %%
data["review_length"] = data["cleaned_tweets"].apply(len)
test_data["review_length"] = test_data["cleaned_tweets"].apply(len)

# %%
sns.distplot(a=data["review_length"])

#%%
sns.distplot(a=test_data["review_length"])

#%%
X = data["cleaned_tweets"].values
y = data["airline_sentiment"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# %%
# one hot encoding target
onehotencoder = OneHotEncoder()
onehotencoder.fit(X=y_train.reshape(-1, 1))

y_train_encoded = onehotencoder.transform(y_train.reshape(-1, 1))
y_test_encoded = onehotencoder.transform(y_test.reshape(-1, 1))
#%%
# Instantiating tokenizer
max_words = 10000
max_len = 80
tok = Tokenizer(num_words=max_words, oov_token="<OOV>")

# %%
tok.fit_on_texts(X_train)
word_index = tok.word_index

vocab_size = len(word_index) + 1

# %%
# Creating train & validation sequence
train_sequences = tok.texts_to_sequences(X_train)
train_padded_sequences = pad_sequences(sequences=train_sequences, maxlen=max_len)

val_sequences = tok.texts_to_sequences(X_test)
val_padded_sequences = pad_sequences(sequences=val_sequences, maxlen=max_len)

#%%
# Creating test sequence
test_sequences = tok.texts_to_sequences(test_data["cleaned_tweets"])
test_padded_sequences = pad_sequences(sequences=test_sequences, maxlen=max_len)

#%%
# Defining shape of input and output & Reguralization Parameter
INPUT_SHAPE = train_padded_sequences.shape[1]
OUT_SHAPE = y_train_encoded.shape[1]
DROPOUT_RATE = 0.3
L1_PENALTY = 0.0001
L2_PENALTY = 0.0001

#%%
# Creating embedding layer
with open("../input/glove.6B.50d.txt") as glove:
    embedding_dict = {}
    for x in glove.readlines():
        word_vectors = x.split()
        word = word_vectors[0]
        embedding_dict[word] = np.asarray(word_vectors[1:], dtype="float32")

print("Loaded %s word vectors." % len(embedding_dict))

#%%
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in word_index.items():
    embedding_vector = embedding_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# %%
# Instantiating sequential model
model = Sequential()
# model.add(Dense(input_shape=(INPUT_SHAPE,), units=max_len))
model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=50,
        input_length=INPUT_SHAPE,
        weights=[embedding_matrix],
        trainable=False,
    )
)
model.add(BatchNormalization())

# Adding hidden layers
model.add(
    Dense(
        units=32,
        activation="relu",
        kernel_initializer="uniform",
        kernel_regularizer=regularizers.l1_l2(l1=L1_PENALTY, l2=L2_PENALTY),
    )
)
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

model.add(
    Dense(
        units=16,
        activation="relu",
        kernel_initializer="uniform",
        kernel_regularizer=regularizers.l1_l2(l1=L1_PENALTY, l2=L2_PENALTY),
    )
)
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

model.add(
    Dense(
        units=8,
        activation="relu",
        kernel_initializer="uniform",
        kernel_regularizer=regularizers.l1_l2(l1=L1_PENALTY, l2=L2_PENALTY),
    )
)
model.add(BatchNormalization())
model.add(Dropout(DROPOUT_RATE))

model.add(Flatten())

# Adding output layer
model.add(Dense(OUT_SHAPE, activation="softmax"))

# %%
# Instantiating early stopping
early_stop = EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")
save_best = ModelCheckpoint(
    filepath="../models/best_model.h5", save_best_only=True, monitor="val_loss"
)

# %%
# Instantiating optimizer
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# %%
model.compile(optimizer=adam, metrics=["accuracy"], loss="categorical_crossentropy")
model.summary()
# %%
model.fit(
    x=train_padded_sequences,
    y=y_train_encoded,
    epochs=100,
    verbose=1,
    callbacks=[early_stop, save_best],
    batch_size=32,
    validation_data=(val_padded_sequences, y_test_encoded),
)

# %%
y_pred = model.predict(val_padded_sequences)

y_pred = np.argmax(y_pred, axis=1)

val_accuracy = accuracy_score(y_test, y_pred)

# %%

airline_sentiment = model.predict(test_padded_sequences)
test_data["airline_sentiment"] = np.argmax(airline_sentiment, axis=1)
# %%
test_data[["tweet_id", "airline_sentiment"]].to_csv(
    "../output/submission_1.csv", index=False
)

# %%
