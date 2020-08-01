#%%
import pandas as pd
import numpy as np

#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from numpy.random import seed
from tensorflow_core import metrics

seed(1)
tf.random.set_seed(2)

#%%
# Setting training environments and other variables
train_batch_size = 64
epochs = 5
latent_dim = 256
num_samples = 10000

# %%
data = pd.read_csv("../input/fra.txt", delimiter="\t", header=None)

# %%
data.rename({0: "English", 1: "French", 2: "Temp"}, inplace=True, axis=1)

# %%
data[["English", "French"]]

# %%
input_texts = []
target_texts = []
input_chars = set()
target_chars = set()

# %%
lines = open("../input/fra.txt", encoding="utf-8").read().split("\n")

# %%
for line in lines:
    try:
        if len(line) > 1:
            input_text, target_text, _ = line.split("\t")
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)

            # Create a set of all unique input characters
            for char in input_text:
                if char not in input_chars:
                    input_chars.add(char)
                    print(char)

            # Create a set of all unique output characters
            for char in target_text:
                if char not in target_chars:
                    target_chars.add(char)
    except:
        print(f"Exception thrown in line {line}")


# %%
print(
    len(input_texts), len(target_texts), len(input_chars), len(target_chars),
)

# %%
input_chars = sorted(list(input_chars))
target_chars = sorted(list(target_chars))
num_encoder_tokens = len(
    input_chars
)  # aka size of the english alphabet + numbers, signs, etc.
num_decoder_tokens = len(
    target_chars
)  # aka size of the french alphabet + numbers, signs, etc.

#%%
input_token_index = {char: i for i, char in enumerate(input_chars)}
target_token_index = {char: i for i, char in enumerate(target_chars)}

#%%
max_encoder_seq_length = max(
    [len(txt) for txt in input_texts]
)  # Get longest sequences length for input
max_decoder_seq_length = max(
    [len(txt) for txt in target_texts]
)  # Get longest sequences length for target

print(max_encoder_seq_length, max_decoder_seq_length)
#%%
# encoder_input_data is a 3D array of shape (num_pairs, max_english_sentence_length, num_english_characters)
# containing a one-hot vectorization of the English sentences.

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
#%%
# decoder_input_data is a 3D array of shape (num_pairs, max_french_sentence_length, num_french_characters)
# containg a one-hot vectorization of the French sentences.

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
#%%
# decoder_target_data is the same as decoder_input_data but offset by one timestep.
# decoder_target_data[:, t, :] will be the same as decoder_input_data[:, t + 1, :]

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
#%%
# Loop over input texts
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    # Loop over each char in an input text
    for t, char in enumerate(input_text):
        # Create one hot encoding by setting the index to 1
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    # Loop over each char in the output text
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# %%
print("Done preparing data for training")


# %%
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens), name="encoder_inputs")

# The return_state contructor argument, configuring a RNN layer to return a list
# where the first entry is the outputs and the next entries are the internal RNN states.
# This is used to recover the states of the encoder.
encoder = LSTM(latent_dim, return_state=True, name="encoder")

encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens), name="decoder_inputs")

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(
    latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
)

# The inital_state call argument, specifying the initial state(s) of a RNN.
# This is used to pass the encoder states to the decoder as initial states.
# Basically making the first memory of the decoder the encoded semantics
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation="softmax", name="decoder_dense")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# %%
# Run training
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()
#%%
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=1000,
    epochs=epochs,
    validation_split=0.2,
)
# Save model
# model.save('s2s.h5')



# %%
