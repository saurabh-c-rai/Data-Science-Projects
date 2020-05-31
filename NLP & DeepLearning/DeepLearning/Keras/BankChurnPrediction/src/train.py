#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
import warnings

warnings.filterwarnings("ignore")

#%%
# load data
path = "../input/Churn_Modelling.csv"
data = pd.read_csv(path)

#%%
# separate into features and target
X = data[
    [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
]
y = data["Exited"]

#%%
# mean normalization and scaling
mean, std = np.mean(X), np.std(X)
X = (X - mean) / std
X = pd.concat(
    [X, pd.get_dummies(data["Gender"], prefix="Gender", drop_first=True)], axis=1
)

#%%
# transform data according to the model input format
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=9, stratify=y
)

#%%
# one-hot encode labels
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


# %%
# Instantiating a sequential model : Output of previous layer is input to current layer
model = Sequential()

# %%
# Creating the input layer
model.add(Dense(64, input_shape=(9,)))


# %%
# activation function for first layer
model.add(Activation("relu"))


# %%
# Creating second layer with activation
model.add(Dense(256, activation="relu"))

#%%
# Creating third layer with activation
model.add(Dense(32, activation="relu"))

# %%
# Creating output layer with activation
model.add(Dense(2, activation="softmax"))

# %%
# Summary of the model
model.summary()

# %%
# Compiling model with SGD optimizer, categorical_cross_entrophy & accuracy metrics
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])


# %%
# Defining callback with Early stop
early_stop = EarlyStopping(patience=5)

# %%
history = model.fit(
    x=X_train, y=y_train, validation_split=0.1, epochs=20, callbacks=[early_stop]
)

#%%
print(history.history.keys())
# %%
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
# Evaulating the model on test set
test_loss, test_acc = model.evaluate(x=X_test, y=y_test)
print(test_loss, test_acc)


# %%
