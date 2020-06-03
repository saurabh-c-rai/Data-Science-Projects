# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df = pd.read_csv(path)
df.head()
X = df.drop(['insuranceclaim'],1)
y = df['insuranceclaim']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 6) 
# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
plt.boxplot(X_train['bmi'])
q_value=X_train['bmi'].quantile(0.95)
q_value

# Code ends here


# --------------
# Code starts here
relation = X_train.corr()
print(relation)
import seaborn as sns
sns.pairplot(X_train)
# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols = ['children','sex','region','smoker']
fig, axes = plt.subplots(2,2)
for i in range(2):
    for j in range(2):
        col = cols[i*2+j]
        sns.countplot(x = X_train[col],hue= y_train,ax=axes[i,j])
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr = LogisticRegression()
grid = GridSearchCV(estimator = lr,param_grid=parameters)
grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
accuracy

# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import numpy as np
# Code starts here
score = roc_auc_score(y_test,y_pred)
y_pred_proba = np.array(pd.DataFrame(grid.predict_proba(X_test)).iloc[:,1])
fpr = metrics.roc_curve(y_test,y_pred)
tpr = metrics.roc_curve(y_test,y_pred_proba)
roc_auc = roc_auc_score(y_test,y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))


# Code ends here


