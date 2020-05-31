#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%%
path = "../input/bank_customer_churn.csv"
data = pd.read_csv(path)

# %%
data.describe()

# %%
data.head()

# %%
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

# %%

# mean normalization and scaling
mean, std = np.mean(X), np.std(X)
X = (X - mean) / std
X = pd.concat(
    [X, pd.get_dummies(data["Gender"], prefix="Gender", drop_first=True)], axis=1
)


# %%
# transform data according to the model input format
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=9
)


# %%
# size of layers
INPUT_SIZE = X_train.shape[1]
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1
np.random.seed(22)

# %%
# Initializing weights
w_1 = np.ones(shape=(INPUT_SIZE, HIDDEN_SIZE)) * 0.05
w_2 = np.ones(shape=(HIDDEN_SIZE, OUTPUT_SIZE)) * 0.05

# %%
# Initializing bias values
b_1 = np.zeros(shape=(1, HIDDEN_SIZE))
b_2 = np.zeros(shape=(1, OUTPUT_SIZE))

# %%
def sigmoid(x):
    """Function to return sigmoid value
    
    Arguments:
        x {[numpy array]} -- [feature vector]
    
    Returns:
        [float] -- [sigmoid equivalent of input]
    """

    return 1 / (1 + np.exp(-(x)))


# %%
def forward(w_1, w_2, b_1, b_2, X):
    """Function to stimulate the forward propogation of Neural Network 
    
    Arguments:
        w_1 {[numpy array]} -- [weight vector from input to hidden layer]
        w_2 {[numpy array]} -- [weight vector from hidden layer to output layer]
        b_1 {[numpy array]} -- [bias vector for the hidden layer]
        b_2 {[numpy array]} -- [bias vector for the hidden layer]
        x {[numpy array]} -- [feature vector row]

    Returns:
        a1 {[numpy array]} -- [post activation value of the hidden layer]
        a2 {[numpy array]} -- [post activation value of the output layer]

    """
    z1 = X @ w_1 + b_1
    a1 = sigmoid(z1)
    z2 = a1 @ w_2 + b_2
    a2 = sigmoid(z2)
    return a1, a2


# %%
_, pred = forward(w_1, w_2, b_1, b_2, X_train)

# %%
pred

# %%
# function for loss function
def cross_entropy(y_actual, y_hat):
    """Method to calculate Cross Entropy loss function value
    
    Arguments:
        y_actual {[Numpy Array]} -- [Actual Target value]
        y_hat {[Numpy Array]} -- [Predicted Target value]
    
    Returns:
        [float] -- [Value of loss function]
    """
    return (1 / y_hat.shape[0]) * np.sum(
        -np.multiply(y_actual.values.reshape(-1, 1), np.log(y_hat))
        - np.multiply((1 - y_actual.values.reshape(-1, 1)), np.log(1 - y_hat))
    )


# %%

# function to score on unseen data
def predict(X_test, y_test):
    """Function to calculate prediction for the given test data.
    Calculates updated weights using backpropagation and then makes prediction on them using forward propogation 
    
    Arguments:
        X_test {[Numpy array/dataframe]} -- [Test set features]
        y_test {[Numpy Array]} -- [Training set Target]

    
    Returns:
        acc[float] -- [Accuracy score for the prediction]
    """

    # finding best set of weights
    w1_new, w2_new, b1_new, b2_new = backpropagate(
        w_1, w_2, b_1, b_2, X_train, X_test, y_train, y_test, 4000, 0.01
    )

    # make predictions
    y_pred = forward(w1_new, w2_new, b1_new, b2_new, X_test.values)[1].flatten()

    # binarize it
    y_pred = y_pred > 0.5

    # calculate accuracy
    acc = accuracy_score(y_pred, y_test.values)

    return acc


# %%
def backpropagate(w_1, w_2, b_1, b_2, X_train, X_test, y_train, y_test, epochs, lr):
    """
    Backpropagation algorithm to find the best set of 
    weights for our problem statement.
    
    Arguments:
        w_1 {[Numpy array]} -- [weights from input to hidden layer]
        w_2 {[Numpy array]} -- [weights from hidden to output layer]
        b_1 {[Numpy array]} -- [bias from input to hidden layer]
        b_2 {[Numpy array]} -- [bias from hidden to output layer]
        X_train {[Numpy array/dataframe]} -- [Training set features]
        X_test {[Numpy array/dataframe]} -- [Test set features]
        y_train {[Numpy Array]} -- [Training set Target]
        y_test {[Numpy Array]} -- [Training set Target]
        epochs {[int]} -- [number of iterations over the entire training data]
        lr {[float]} -- [learning rate]

    Returns:
        w_1[Numpy Array] -- [Updated weights from input to hidden layer]
        w_2[Numpy Array] -- [Updated weights from hidden layer to output layer]
        b_1[Numpy Array] -- [Updated bias from input to hidden layer]
        b_2[Numpy Array] -- [Updated bias from hidden layer to output layer]        
    """
    m = X_train.shape[0]
    for _ in range(epochs):
        a1, a2 = forward(w_1, w_2, b_1, b_2, X_train.values)

        curr_loss = cross_entropy(y_train, a2)

        da2 = a2 - y_train.values.reshape(-1, 1)
        da1 = np.multiply((da2 @ w_2.T), np.multiply(a1, 1 - a1))

        dw2 = (1 / m) * a1.T @ (da2)
        db2 = 1 / m * da2

        dw1 = (1 / m) * X_train.values.T @ (da1)
        db1 = 1 / m * da1

        w_1 = w_1 - lr * dw1
        w_2 = w_2 - lr * dw2

        b_1 -= np.sum(db1)
        b_2 -= np.sum(db2)

    return w_1, w_2, b_1, b_2


#%%
acc = predict(X_test, y_test)


# %%
print(acc)
