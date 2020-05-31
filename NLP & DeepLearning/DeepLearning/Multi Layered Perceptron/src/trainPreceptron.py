#%%
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
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
# Random Initialization of weights & bias
w = np.ones(X.shape[1],)
b = 0

# %%
class Perceptron:
    """Class to demonstrate the Perceptron learning rule
    """

    def __init__(self, X, y, w, b, epochs=1000):
        """Method to intialize the features, target, weights,
        bias and number of iterations
        
        Arguments:
            X {[dataframe]} -- [feature]
            y {[dataframe]} -- [target variable to predict]
            w {[numpy array]} -- [intial values of weights]
            b {[numpy array]} -- [intial values of bias]
        
        Keyword Arguments:
            epochs {int} -- [number of iterations] (default: {1000})
        """
        self.b = b
        self.w = w
        self.X = X
        self.y = y
        self.epochs = epochs

    def learn(self, x):
        """Method to compute dot product between feature vector and weight vector

        Arguments:
            x {[numpy array]} -- [feature vector or row of feature dataframe]
        
        Returns:
            [int] -- [1 if dot product >= bias & 0 if dot product < bias]
        """
        return 1 if np.dot(self.w, x) >= self.b else 0

    def fit(self):
        """Method to train the perceptron
        
        Returns:
            max_accuracy{[float]} -- [maximum accuracy received during training]
            bias{[numpy array]} -- [final bias values for the perceptron]
            weight{[numpy array]} -- [final weight values for the perceptron]
            
        """
        # dictionary to store accuracy values
        accuracy = {}
        # maximum accuracy
        max_accuracy = 0

        for epoch in range(self.epochs):
            # iterate over every data point
            for x, y in zip(self.X, self.y):
                # prediction for data point
                pred = self.learn(x)

                # weight update
                if pred == 0 and y == 1:
                    self.w += x
                    self.b += 1
                elif pred == 1 and y == 0:
                    self.w -= x
                    self.b -= 1

            # store the accuracy according to iternation number
            accuracy[epoch] = accuracy_score(self.predict(self.X), self.y)

            # display if new maximum training accuracy is achieved
            if accuracy[epoch] > max_accuracy:
                print(f"Training accuracy at epoch {epoch} is: {accuracy[epoch]}")
                print("=" * 100)
                max_accuracy = accuracy[epoch]

                # weight and bias for maximum accuracy
                chkpt_w = self.w
                chkpt_b = self.b

            self.b = chkpt_b
            self.w = chkpt_w

        return max_accuracy, self.b, self.w

    def predict(self, test):
        """Function to predict on new data
        
        Arguments:
            test {[dataframe]} -- [Test/validation dataframe]
        
        Returns:
            [numpy array] -- [array of predicted value of target]
        """

        # list to store predictions
        preds = []
        for row in test:
            y_pred = self.learn(row)
            preds.append(y_pred)
        return np.array(preds)

    def accuracy(self, X_test, y_test):
        """Method to calculate accuracy of prediction.
        This method calls the fit and predict function internally.
        Arguments:
            X_test {[dataframe]} -- [test/validation feature dataframe]
            y_test {[dataframe]} -- [actual target values for given feature]
        
        Returns:
            accuracy[float] -- [accuracy of prediction on test/validation data]
        """
        _, self.b, self.w = self.fit()
        preds = self.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        return accuracy


# %%
perceptron = Perceptron(X_train.values, y_train.values, w, b, 1000)

# %%
acc = perceptron.accuracy(X_test.values, y_test.values)

# %%
print(acc)

# %%
