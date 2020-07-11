#%%
#%%
import os
import re
from collections import Counter
from string import punctuation

import keras
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import LSTM, BatchNormalization, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

InteractiveShell.ast_node_interactivity = "all"


#%%
cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

# %%
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# %%
train_data.head().T
test_data.head().T
#%%
train_data.shape

# %%
train_data["Score"].value_counts() / train_data["Score"].value_counts().sum()

# %%
def remove_URL(text):
    """[function to remove URL from the string]

    Arguments:
        text {[String]} -- [Sentence which may or may not contains a URL]

    Returns:
        [string] -- [sentence without any URL]
    """
    url = re.compile(r"https?://\S+ |www\.\S+")
    return url.sub(r" ", text)


#%%
def remove_html(text):
    """[function to remove html tags from the input sentence]

    Arguments:
        text {[String]} -- [Sentence which may or may not contains an html tag]

    Returns:
        [string] -- [sentence without any html tag]
    """
    html = re.compile(r"<.*?>")
    return html.sub(r" ", text)


#%%
def remove_emoji(string):
    """Function to remove emoji from the string

    Arguments:
        string {[String]} -- [Sentence with or without emoji]

    Returns:
        [String] -- [Sentence without any emoji]
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600 - \U0001F64F"  # emoticons
        "\U0001F300 - \U0001F5fF"  # symbols & pictographs
        "\U0001F680 - \U0001F6FF"  # transport & map symbols
        "\U0001F1E0 - \U0001F1FF"  # flags
        "\U00002702 - \U000027B2"
        "\U000024C2 - \U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r" ", string)


#%%
def remove_punct(text):
    """Function to remove punctuations from the sentence

    Arguments:
        text {[String]} -- [Sentences or paragraphs with or without punctuations]

    Returns:
        [String] -- [Sentences or paragraphs without any punctuations]
    """
    table = str.maketrans(" ", " ", punctuation)
    return text.translate(table)


#%%
# Initialize all objects
stopword = set(stopwords.words("english"))
lemma = WordNetLemmatizer()
tokenizer = WordPunctTokenizer()

#%%
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


#%%
train_data["Text"] = train_data["Text"].map(lambda x: remove_URL(x))
train_data["Text"] = train_data["Text"].map(lambda x: remove_html(x))
train_data["Text"] = train_data["Text"].map(lambda x: remove_emoji(x))
train_data["Text"] = train_data["Text"].map(lambda x: remove_punct(x))
train_data["Text"] = train_data["Text"].map(lambda x: preprocess_document(x))

#%%
test_data["Text"] = test_data["Text"].map(lambda x: remove_URL(x))
test_data["Text"] = test_data["Text"].map(lambda x: remove_html(x))
test_data["Text"] = test_data["Text"].map(lambda x: remove_emoji(x))
test_data["Text"] = test_data["Text"].map(lambda x: remove_punct(x))
test_data["Text"] = test_data["Text"].map(lambda x: preprocess_document(x))
#%%
#%%
train_data["Text_length"] = train_data["Text"].apply(len)
test_data["Text_length"] = test_data["Text"].apply(len)

#%%
train_data.head().T
test_data.head().T
#%%
X = train_data[["Text", "Text_length"]]
y = train_data["Score"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("No of train records: {}".format(X_train.shape))
print("No of test records: {}".format(X_test.shape))
#%%
# # Encoding target
# oneHotEncoder = OneHotEncoder()
# oneHotEncoder.fit(y_train.values.reshape(-1, 1))
# y_train_encoded = oneHotEncoder.transform(y_train.values.reshape(-1, 1))
# y_test_encoded = oneHotEncoder.transform(y_test.values.reshape(-1, 1))
#%%
plt.figure(figsize=(12, 6))
sns.distplot(X_train["Text_length"], bins=50)
sns.distplot(X_test["Text_length"], bins=50)
sns.distplot(test_data["Text_length"], bins=50)
plt.show()
#%%
max_review_length_train = np.max(train_data["Text"].str.split().apply(len))
max_review_length_test = np.max(test_data["Text"].str.split().apply(len))


#%%
max_words = 10000
max_length = 600

#%%
tokenizer = Tokenizer(num_words=max_words)

#%%
tokenizer.fit_on_texts(X_train["Text"])

#%%
word_index = tokenizer.word_index

#%%
vocab_size = len(word_index) + 1

#%%
train_sequences = tokenizer.texts_to_sequences(X_train["Text"])
test_sequences = tokenizer.texts_to_sequences(X_test["Text"])
#%%
train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding="post", truncating="post"
)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding="post", truncating="post"
)
#%%
print(X_train["Text"].iloc[0])
print(train_padded[0])

#%%
reverse_word_index = {v: k for (k, v) in word_index.items()}

#%%
def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


#%%
decode(train_sequences[0])

#%%
print(f"Shape of train {train_padded.shape}")
print(f"Shape of test {test_padded.shape}")

#%%
y_train_categorical = to_categorical(y=y_train.values)
y_test_categorical = to_categorical(y=y_test.values)
#%%
y_train_categorical[0:5], y_train.head()

#%%
# Defining shape of input and output & Reguralization Parameter
INPUT_SHAPE = train_padded.shape[1]
OUT_SHAPE = y_train_categorical.shape[1]
DROPOUT_RATE = 0.3
L1_PENALTY = 0.0001
L2_PENALTY = 0.0001
#%%
# DeepLearning model creation
model = Sequential()

# Adding 2 Dense layer and one output layer
model.add(Embedding(vocab_size, 50, input_length=max_length))
model.add(BatchNormalization())
model.add(Dense(units=8, input_dim=INPUT_SHAPE))
model.add(BatchNormalization())
model.add(LSTM(64, dropout=0.1))
model.add(BatchNormalization())
model.add(Dense(6, activation="softmax", kernel_initializer="uniform"))
model.add(BatchNormalization())
#%%
# Creating Stochastic Gradient Descent with time decay learning rate
learning_rate = 0.01
epoch = 20

adam = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

# %%
# compiling model
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
#%%
model.fit(
    x=train_padded,
    y=y_train_categorical,
    batch_size=64,
    epochs=epoch,
    validation_data=(test_padded, y_test_categorical),
)

#%%
# data["Text"] = data["Text"].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
# data["Text"] = data["Text"].apply(lambda x: re.sub("<[^>]*>", " ", x))

# data["Text"] = data["Text"].apply(lambda x: x.lower())

# data["Text"] = [nltk.word_tokenize(x) for x in data["Text"]]


# data["Text"] = data["Text"].apply(
#     lambda row: [word for word in row if word not in stopwords]
# )

# data["Text"] = data["Text"].apply(lambda x: [lemma.lemmatize(i) for i in x])

# data["Text"] = data["Text"].apply(lambda x: " ".join(x))

# data.head()
#%%
tf_id = TfidfVectorizer(ngram_range=(3, 7), max_features=6000)

# %%
tf_id.fit(train_data["Text"])

# %%
X = tf_id.transform(train_data["Text"])

# %%
X.shape

# %%
y = train_data["Score"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("No of train records: {}".format(X_train.shape[0]))
print("No of test records: {}".format(X_test.shape[0]))

# %%
clf_model_tf_idf = RandomForestClassifier(
    verbose=4, n_estimators=1000, class_weight="balanced",
)


# %%
clf_tf_idf = clf_model_tf_idf.fit(X_train, y_train)

#%%
y_pred = clf_tf_idf.predict(X_test)
# %%
f1_score(y_test, y_pred, average="weighted")

#%%
f1_score(y_train, clf_tf_idf.predict(X_train), average="weighted")


# %%
