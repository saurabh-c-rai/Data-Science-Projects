#%%
import re
from collections import Counter
from string import punctuation

import keras
import nltk
import numpy as np
import pandas as pd
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# %%
data = pd.read_csv("../input/train.csv")

# %%
data.head().T
#%%
data.shape

# %%
data["Score"].value_counts() / data["Score"].value_counts().sum()

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
data["Text"] = data["Text"].map(lambda x: remove_URL(x))
data["Text"] = data["Text"].map(lambda x: remove_html(x))
data["Text"] = data["Text"].map(lambda x: remove_emoji(x))
data["Text"] = data["Text"].map(lambda x: remove_punct(x))

#%%
# Stop words removal
stopword = set(stopwords.words("english"))
lemma = WordNetLemmatizer()


def remove_stopwords(text):
    """Function to remove stopwords and change all the words to lower case

    Arguments:
        text {[String]} -- [Sentence or paragraph]

    Returns:
        [String] -- [Sentence or paragraphs without stopwords and all in lower case]
    """
    text = [word.lower() for word in text.split() if word.lower() not in stopword]
    lemmatized_text = [lemma.lemmatize(i) for i in text]
    return " ".join(lemmatized_text)


#%%
data["Text"] = data["Text"].map(remove_stopwords)

#%%
data.head().T

#%%
np.max(data["Text"].str.split().apply(len))


#%%
def counter_word(text):
    count = Counter()
    for i in text.values:
        for word in i.split():
            count[word] += 1
    return count


#%%
counter = counter_word(text=data["Text"])

#%%
num_words = len(counter)

#%%
max_length = 500

#%%
X = data["Text"]

# %%
y = data["Score"]

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("No of train records: {}".format(X_train.shape))
print("No of test records: {}".format(X_test.shape))

#%%
tokenizer = Tokenizer(num_words=num_words)

#%%
tokenizer.fit_on_texts(X_train)

#%%
word_index = tokenizer.word_index

#%%
word_index

#%%
train_sequences = tokenizer.texts_to_sequences(X_train)

#%%
train_padded = pad_sequences(
    train_sequences, maxlen=max_length, padding="post", truncating="post"
)
#%%
print(X_train[0])
print(train_padded[0])

#%%
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#%%
def decode(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


#%%
decode(train_sequences[0])

#%%
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(
    test_sequences, maxlen=max_length, padding="post", truncating="post"
)

#%%
print(f"Shape of train {train_padded.shape}")
print(f"Shape of test {test_padded.shape}")

#%%
y_train_categorical = to_categorical(y=y_train.values)
y_test_categorical = to_categorical(y=y_test.values)
#%%
y_train_categorical[0:5], y_train.head()
#%%
model = Sequential()

#%%
# Adding 2 Dense layer and one output layer
model.add(Embedding(num_words, 32, input_length=max_length))
model.add(LSTM(64, dropout=0.1))
model.add(Dense(6, activation="softmax", kernel_initializer="uniform"))

#%%
# Creating Stochastic Gradient Descent with time decay learning rate
learning_rate = 0.01
epoch = 100
decay_rate = 0.000001
momentum = 0.9
opt_time_decay = optimizers.SGD(
    lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False
)

# %%
# compiling model
model.compile(
    optimizer=opt_time_decay, loss="categorical_crossentropy", metrics=["accuracy"]
)
#%%
model.fit(
    x=train_padded,
    y=y_train_categorical,
    epochs=20,
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
tf_id.fit(data["Text"])

# %%
X = tf_id.transform(data["Text"])

# %%
X.shape

# %%
y = data["Score"]

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
