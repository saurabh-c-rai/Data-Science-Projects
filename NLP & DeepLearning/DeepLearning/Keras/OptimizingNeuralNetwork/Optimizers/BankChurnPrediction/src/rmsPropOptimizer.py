#%%
# import packages
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.layers import Dropout
from keras import regularizers
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    Callback,
)
import math

#%%
# load data
path = "../input/bankcustomerchurn.csv"
dataset = pd.read_csv(path)

#%%
# split into features and target
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#%%
# creating label encoder object no. 1 to encode region name(index 1 in features)
labelencoder_X_1 = LabelEncoder()

# encoding region from string to just 3 no.s 0,1,2 respectively
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#%%
# creating label encoder object no. 2 to encode Gender name(index 2 in features)
labelencoder_X_2 = LabelEncoder()

# encoding Gender from string to just 2 no.s 0,1(male,female) respectively
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#%%
# OneHot encoding using OneHotEncoder
columnTransform = ColumnTransformer(
    [("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = columnTransform.fit_transform(X)
X = X[:, 1:]

#%%
# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

#%%
# transform train and test features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%
# Size of input & output layer
INPUT_LAYER = X_train.shape[1]
OUTPUT_LAYER = np.unique(y_train).shape[0]

#%%
# Instantiate keras sequential model
model_5 = Sequential()

#%%
# Adding 2 Dense layer and one output layer
model_5.add(
    Dense(
        activation="relu",
        input_shape=(INPUT_LAYER,),
        units=6,
        kernel_initializer="uniform",
    )
)
model_5.add(
    Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")
)

model_5.add(
    Dense(activation="relu", input_dim=11, units=10, kernel_initializer="uniform")
)

model_5.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

#%%
save_best = ModelCheckpoint(
    filepath=f"../model/best_model.h5", monitor="val_loss", save_best_only=True
)

#%%
class CustomCallbackClass(Callback):
    """[Custom callback class to implement callback]

    Arguments:
        Callback {[Keras Class]} -- [Abstract base class to customize callback]
    """

    def on_train_begin(self, logs=None):
        # keys = list(logs.keys())
        print(f"Starting training")

    def on_train_end(self, logs=None):
        # keys = list(logs.keys())
        print(f"Stop training")

    def on_epoch_begin(self, epoch, logs=None):
        # keys = list(logs.keys())
        print(f"Start epoch {epoch} of training")

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"End epoch {epoch} of training; got log keys: {keys}")

    def on_test_begin(self, logs=None):
        # keys = list(logs.keys())
        print(f"Start testing")

    def on_test_end(self, logs=None):
        # keys = list(logs.keys())
        print(f"Stop testing")

    def on_predict_begin(self, logs=None):
        # keys = list(logs.keys())
        print(f"Start predicting")

    def on_predict_end(self, logs=None):
        # keys = list(logs.keys())
        print(f"Stop predicting")

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Training: start of batch {batch}; got log keys: {keys}")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Training: end of batch {batch}; got log keys: {keys}")

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Evaluating: start of batch {batch}; got log keys: {keys}")

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Evaluating: end of batch {batch}; got log keys: {keys}")

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Predicting: start of batch {batch}; got log keys: {keys}")

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"...Predicting: end of batch {batch}; got log keys: {keys}")


customCallback = CustomCallbackClass()


#%%
# Creating Stochastic Gradient Descent with time decay learning rate
rmsprop = optimizers.RMSprop(learning_rate=0.01, rho=0.9, decay=0.01)

# %%
# compiling model
model_5.compile(optimizer=rmsprop, loss="binary_crossentropy", metrics=["accuracy"])

# %%
# fitting the model on training set
model_5.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=100,
    verbose=1,
    callbacks=[save_best],
    validation_data=(X_test, y_test),
)

# %%
# evaluating on unknown data
score = model_5.evaluate(x=X_test, y=y_test)

# %%
print(score[1])
# %%
