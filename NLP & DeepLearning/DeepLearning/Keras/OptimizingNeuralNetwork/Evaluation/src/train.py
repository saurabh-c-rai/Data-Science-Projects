#%%
# import modules
import keras
import numpy as np
import pandas as pd
import seaborn as sns
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

#%%
# Load the train data
train_data = pd.read_csv("../input/train.csv")
train_data.head()

#%%
train_data.isna().sum()

#%%
train_data.drop(columns=["cust_id"], inplace=True)

#%%
X = train_data.iloc[:, :-1].copy()
y = train_data.iloc[:, -1].copy()

#%%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=3
)

#%%
scalar = StandardScaler()
scalar.fit(X_train)

#%%
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

#%%
y_train.value_counts() / y_train.value_counts().sum()
y_test.value_counts() / y_test.value_counts().sum()

#%%
oneHotEncoder = OneHotEncoder()
oneHotEncoder.fit(y_train.values.reshape(-1, 1))

#%%
y_train = oneHotEncoder.transform(y_train.values.reshape(-1, 1))
y_test = oneHotEncoder.transform(y_test.values.reshape(-1, 1))

#%%
# Input Layer size and Output layer size
INPUT_SIZE = X_train.shape[1]
OUTPUT_SIZE = y_train.shape[1]

#%%
model = Sequential()

#%%
# Input Layer
model.add(Dense(input_shape=(INPUT_SIZE,), activation="relu", units=64))
model.add(Dropout(0.1))
model.add(BatchNormalization())

#%%
# Layer 1 & 2
model.add(Dense(input_dim=INPUT_SIZE, activation="relu", units=32))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(input_dim=INPUT_SIZE, activation="relu", units=16))
model.add(BatchNormalization())
model.add(Dropout(0.1))

#%%
# Output layer
model.add(Dense(OUTPUT_SIZE, activation="softmax"))

#%%
# Defining callback with Early stop
early_stop = EarlyStopping(patience=10)

save_best = ModelCheckpoint(
    filepath=f"../model/best_model.h5", monitor="val_loss", save_best_only=True
)
#%%
opt = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

# %%
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])


# %%
# fitting model
model.fit(
    x=X_train,
    y=y_train,
    batch_size=256,
    epochs=100,
    verbose=1,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, save_best],
)


# %%
pd.DataFrame(model.history.history).plot(figsize=(10, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# %%
test_data = pd.read_csv("../input/test.csv")

# %%
test_df = test_data.copy()

# %%
test_data.iloc[:, 1:] = scalar.transform(test_data.iloc[:, 1:])

# %%
test_prediction = model.predict(test_data.iloc[:, 1:])

# %%
test_data["loan_status"] = np.argmax(test_prediction, axis=1)

# %%
test_data["loan_status"].value_counts()

# %%
test_data[["cust_id", "loan_status"]].to_csv(
    "../output/optimized_prediction.csv", index=False
)
# %%
