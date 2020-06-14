#%%
from string import punctuation

import keras.backend as K
import mlflow.keras
import nltk
import numpy as np
import pandas as pd
import mlflow

# visualize model structure
from IPython.display import SVG
from keras.layers import Dense, Embedding, Lambda
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
from nltk.corpus import gutenberg

mlflow.keras.autolog()
#%%
# Loading the data
corpus = [nltk.corpus.gutenberg.words("shakespeare-caesar.txt")]

print("Sample Corpus(Julius Caesar- Shakespeare):\n")
print(corpus[0][16:])
print(corpus[0][150:])

#%%
# Terms to remove
r_terms = punctuation + "1234567890"
#%%
# Removing the above defined terms
norm_corpus = [
    [word.lower() for word in sent if word not in r_terms] for sent in corpus
]
#%%
# Joining the sentence back
norm_corpus = [" ".join(tok_sent) for tok_sent in norm_corpus]
#%%
# Tokenizing the words
norm_corpus = [tok_sent for tok_sent in norm_corpus if len(tok_sent.split()) > 2]
norm_corpus = norm_corpus[:1500]
#%%
# Tokenizing parameters
embedding_dim = 100
trunc_type = "post"
oov_tok = "<OOV>"


tokenizer = Tokenizer(oov_token=oov_tok)


tokenizer.fit_on_texts(norm_corpus)
word2index = tokenizer.word_index
#%%
# Building the vocabulary of unique words
word2index["PAD"] = 0
index2word = {v: k for k, v in word2index.items()}
sequences = tokenizer.texts_to_sequences(norm_corpus)

# Choosing the vocabulary size and context window size
vocab_size = len(word2index)
window_size = 2  # context window size

print("\nVocabulary Size:", vocab_size)
print("\nEmbedding Size:", embedding_dim)
print("\nVocabulary Sample:\n", list(word2index.items())[:10])
#%%
def gcp(corpus, vocab_size, window_size):
    """Function for generating context word pairs.

    Args:
        corpus ([array]): [Array of corpus converted to sequence]
        vocab_size ([int]): [length of vocabulary]
        window_size ([type]): [context window]

    Yields:
        x ([Numpy array]: [Padded sequence]
        y ([Numpy array]): [One hot encoded target]
    """
    # We take context length as double the window size to include both the left and right parts

    context_len = window_size * 2

    for words in corpus:

        sentence_len = len(words)

        for index, word in enumerate(words):
            context_words = []
            label_word = []
            start = index - window_size
            end = index + window_size + 1

            context_words.append(
                [
                    words[i]
                    for i in range(start, end)
                    if 0 <= i < sentence_len and i != index
                ]
            )
            label_word.append(word)

            x = pad_sequences(context_words, maxlen=context_len)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)


# Sample test for 10 samples
now = 0
for x, y in gcp(corpus=sequences, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print(
            "X (Context Words):",
            [index2word[w] for w in x[0]],
            "; Y(Target Word):",
            index2word[np.argwhere(y[0])[0][0]],
        )

        if now == 5:
            break
        now += 1

#%%
# build CBOW architecture
cbow = Sequential()
cbow.add(
    Embedding(
        input_dim=vocab_size, output_dim=embedding_dim, input_length=window_size * 2
    )
)
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
cbow.add(Dense(vocab_size, activation="softmax"))
cbow.compile(loss="categorical_crossentropy", optimizer="rmsprop")

# view model summary
print(cbow.summary())

#%%
SVG(
    model_to_dot(cbow, show_shapes=True, show_layer_names=False, rankdir="TB").create(
        prog="dot", format="svg"
    )
)

# %%
for epoch in range(1, 6):
    loss = 0.0
    i = 0
    for x, y in gcp(corpus=sequences, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100000 == 0:
            print("Processed {} (context, word) pairs".format(i))

    print("Epoch:", epoch, "\tLoss:", loss)


weights = cbow.get_weights()[0]
weights = weights[1:]
print(weights.shape)


cbow = pd.DataFrame(weights, index=list(index2word.values())[1:])

print(cbow.head(5))


# %%
