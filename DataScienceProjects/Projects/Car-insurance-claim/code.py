# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
#print(df.head())
#print(df.info())
df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]

for i in df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].columns:
    df[i]= df[i].str.replace('$','')
    df[i] = df[i].str.replace(',','')

X = df.drop('CLAIM_FLAG',1)
y = df['CLAIM_FLAG']

count = y.value_counts()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 ,random_state = 6)
df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]

# Code ends here


# --------------
# Code starts here
X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]= X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype('float')
X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype('float')

X_train.isnull().sum()
X_test.isnull().sum()
# Code ends here


# --------------
# Code starts here

X_train=X_train.dropna(subset=['YOJ','OCCUPATION'])
X_test=X_test.dropna(subset=['YOJ','OCCUPATION'])

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
for i in X_train[['AGE','CAR_AGE','INCOME','HOME_VAL']].columns:
    X_train[i] = X_train[i].fillna(X_train[i].mean())
    X_test[i] = X_test[i].fillna(X_test[i].mean())

X_train.isnull().sum()
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le = LabelEncoder()
X_train[columns].dtypes
for i in columns:
    X_train[i]=X_train[i].astype('str')
    X_train[i]=le.fit_transform(X_train[i])
    X_test[i]=X_test[i].astype('str')
    X_test[i]=le.transform(X_test[i])

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model=LogisticRegression(random_state = 6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)
precision=precision_score(y_test,y_pred)
print(precision)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=6)
X_train,y_train=smote.fit_sample(X_train,y_train)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
print(score)


# Code ends here


