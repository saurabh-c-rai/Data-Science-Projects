#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %%
# Reading the training set
df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

# %%
# First five rows
df.head()

# %%
# shape for training set
df.shape

#%%
# Splitting into features and target
X_train = df.iloc[:, :-1].copy()
y_train = df.iloc[:, -1].copy()

#%%
X_test = test_df.copy()

#%%
X_test.head()

#%%
X_train.head()

#%%
y_train_categorical = to_categorical(y=y_train)

#%%
y_train_categorical.shape
#%%
scalar = MinMaxScaler()

X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

#%%
X_train.shape

#%%
X_test.shape

#%%
# Size of input & output layer
INPUT_LAYER = X_train.shape[1]
OUTPUT_LAYER = y_train.nunique()
# %%
model = Sequential()

# %%
model.add(Dense(32, input_shape=(INPUT_LAYER,)))
model.add(Activation("relu"))

# %%
model.add(Dense(64, activation="relu"))
model.add(Dense(128, activation="tanh"))
model.add(Dense(10, activation="softmax"))


# %%
model.summary()

# %%
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

#%%
# Implementing callback
save_best = ModelCheckpoint(f"../model/best_model.h5", save_best_only=True)
early_stop = EarlyStopping(patience=5)

# %%
history = model.fit(
    x=X_train,
    y=y_train_categorical,
    epochs=20,
    validation_split=0.2,
    callbacks=[save_best, early_stop],
)

#%%
# summarize history for accuracy
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
#%%
# summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# %%
y_pred = model.predict(X_test)


# %%
y_pred

# %%
y_pred_labels = np.argmax(y_pred, axis=1)

#%%
y_pred_labels

# %%
pd.DataFrame(y_pred_labels).to_csv("../output/prediction.csv", index=False)


# %%
pred = pd.DataFrame(y_pred_labels)

# %%
pred[0].value_counts().sum()

# %%
pred[0] = pred[0] + 1

#%%
y_train.value_counts()

# %%
pred[0]

# %%
