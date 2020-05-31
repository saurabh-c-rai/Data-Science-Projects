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
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
# OneHot encoding using ColumnTransformer
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
model_2 = Sequential()

#%%
# Adding 2 Dense layer and one output layer
model_2.add(Dense(activation="relu", input_shape=(INPUT_LAYER,), units=6))
model_2.add(Dense(activation="relu", input_dim=11, units=6))
model_2.add(Dropout(rate=0.1))
model_2.add(Dense(1, activation="sigmoid"))

#%%
# Regularizing using EarlyStopping method
early_stop = EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)

#%%
# Creating a Stochastic Gradient Descent optimizer
opt = optimizers.SGD(lr=0.01, momentum=0, nesterov=False)

# %%
# compiling model
model_2.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

# %%
# fitting the model on training set
model_2.fit(
    x=X_train, y=y_train, batch_size=32, epochs=100, verbose=1, callbacks=[early_stop]
)

# %%
# evaluating on unknown data
score = model_2.evaluate(x=X_test, y=y_test)

# %%
print(score[1])
# %%
