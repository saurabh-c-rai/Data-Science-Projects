#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#%%
import os

cd_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(cd_path)

# %%
df = pd.read_csv("../tmp/train.csv", keep_default_na=False)
#%%
X_train, X_val, y_train, y_val = train_test_split(
    np.array(df["text"]),
    np.array(df["summary"]),
    test_size=0.1,
    random_state=0,
    shuffle=True,
)


# %%
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#%%
# prepare a tokenizer for reviews on training data

x_tokenizer = Tokenizer(lower=False)
#%%
x_tokenizer.fit_on_texts(X_train)

#%%
thresh = 5

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if value < thresh:
        cnt = cnt + 1
        freq = freq + value

#%%
max_text_len = 314
max_summary_len = 27
# %%
# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
x_tokenizer.fit_on_texts(list(X_train))

#%%
# convert text sequences into integer sequences
x_tr_seq = x_tokenizer.texts_to_sequences(X_train)
x_val_seq = x_tokenizer.texts_to_sequences(X_val)

#%%
# padding zero upto maximum length
x_tr = pad_sequences(x_tr_seq, maxlen=max_text_len, padding="post")
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding="post")

#%%
# size of vocabulary ( +1 for padding token)
x_voc = x_tokenizer.num_words + 1

# %%
# prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))

#%%
thresh = 6

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if value < thresh:
        cnt = cnt + 1
        freq = freq + value

#%%
# prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer(num_words=tot_cnt - cnt)
y_tokenizer.fit_on_texts(list(y_train))

#%%
# convert text sequences into integer sequences
y_tr_seq = y_tokenizer.texts_to_sequences(y_train)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

#%%
# padding zero upto maximum length
y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding="post")
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding="post")

#%%
# size of vocabulary
y_voc = y_tokenizer.num_words + 1

# %%
# Finding rows with only two words
ind = []
for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

#%%
y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)

#%%
ind = []
for i in range(len(y_val)):
    cnt = 0
    for j in y_val[i]:
        if j != 0:
            cnt = cnt + 1
    if cnt == 2:
        ind.append(i)

#%%
y_val = np.delete(y_val, ind, axis=0)
x_val = np.delete(x_val, ind, axis=0)
#%%
# Attention class (Can be ignored to understand)
import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

#%%
class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(
            name="W_a",
            shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
            initializer="uniform",
            trainable=True,
        )
        self.U_a = self.add_weight(
            name="U_a",
            shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
            initializer="uniform",
            trainable=True,
        )
        self.V_a = self.add_weight(
            name="V_a",
            shape=tf.TensorShape((input_shape[0][2], 1)),
            initializer="uniform",
            trainable=True,
        )

        super(AttentionLayer, self).build(
            input_shape
        )  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print("encoder_out_seq>", encoder_out_seq.shape)
            print("decoder_out_seq>", decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(
                states, type(states)
            )
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(
                K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden)
            )
            if verbose:
                print("wa.s>", W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(
                K.dot(inputs, self.U_a), 1
            )  # <= batch_size, 1, latent_dim
            if verbose:
                print("Ua.h>", U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(
                K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden))
            )
            if verbose:
                print("Ws+Uh>", reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print("ei>", e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print("ci>", c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(
                fake_state, [1, hidden_size]
            )  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[1]
        )  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(energy_step, decoder_out_seq, [fake_state_e],)

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(context_step, e_outputs, [fake_state_c],)

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1])),
        ]


# %%
from tensorflow.keras.layers import (
    LSTM,
    Bidirectional,
    Embedding,
    Concatenate,
    TimeDistributed,
    Dense,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

K.clear_session()
#%%
latent_dim = 300
embedding_dim = 100

#%%
# Encoder
encoder_inputs = Input(shape=(max_text_len,))

# embedding layer
enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)

# encoder lstm 1
encoder_lstm1 = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True,
    dropout=0.4,
    recurrent_dropout=0.4,
)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# encoder lstm 2
encoder_lstm2 = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True,
    dropout=0.4,
    recurrent_dropout=0.4,
)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# encoder lstm 3
encoder_lstm3 = LSTM(
    latent_dim,
    return_state=True,
    return_sequences=True,
    dropout=0.4,
    recurrent_dropout=0.4,
)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)
#%%
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

# embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True,
    dropout=0.4,
    recurrent_dropout=0.2,
)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
    dec_emb, initial_state=[state_h, state_c]
)

#%%
# Attention layer
attn_layer = AttentionLayer(name="attention_layer")
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

#%%
# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name="concat_layer")(
    [decoder_outputs, attn_out]
)

# dense layer
decoder_dense = TimeDistributed(Dense(y_voc, activation="softmax"))
decoder_outputs = decoder_dense(decoder_concat_input)
#%%
# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()


# %%
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=2)
modelcheckpoint = ModelCheckpoint(
    "../models/best_model.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=False,
    save_weights_only=False,
    mode="auto",
    save_freq="epoch",
)
#%%
history = model.fit(
    [x_tr, y_tr[:, :-1]],
    y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
    epochs=50,
    callbacks=[es, modelcheckpoint],
    batch_size=128,
    validation_data=(
        [x_val, y_val[:, :-1]],
        y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:],
    ),
)


# %%
from matplotlib import pyplot

pyplot.plot(history.history["loss"], label="train")
pyplot.plot(history.history["val_loss"], label="test")
pyplot.legend()
pyplot.show()
#%%
reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

#%%
# Encode the input sequence to get the feature vector
encoder_model = Model(
    inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c]
)

# Decoder setup
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)
# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c]
)

# attention inference
attn_out_inf, attn_states_inf = attn_layer(
    [decoder_hidden_state_input, decoder_outputs2]
)
decoder_inf_concat = Concatenate(axis=-1, name="concat")(
    [decoder_outputs2, attn_out_inf]
)

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
    [decoder_inputs]
    + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2],
)

#%%
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index["start"]

    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != "end":
            decoded_sentence += " " + sampled_token

        # Exit condition: either hit max length or find stop word.
        if sampled_token == "end" or len(decoded_sentence.split()) >= (
            max_summary_len - 1
        ):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


#%%
def seq2summary(input_seq):
    newString = ""
    for i in input_seq:
        if (i != 0 and i != target_word_index["start"]) and i != target_word_index[
            "end"
        ]:
            newString = newString + reverse_target_word_index[i] + " "
    return newString


def seq2text(input_seq):
    newString = ""
    for i in input_seq:
        if i != 0:
            newString = newString + reverse_source_word_index[i] + " "
    return newString


#%%
for i in range(0, 100):
    print("Review:", seq2text(x_tr[i]))
    print("Original summary:", seq2summary(y_tr[i]))
    print("Predicted summary:", decode_sequence(x_tr[i].reshape(1, max_text_len)))
    print("\n")

