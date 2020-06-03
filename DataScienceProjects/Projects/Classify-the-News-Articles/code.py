# --------------
# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix


# Code starts here
news = pd.read_csv(path)
news = news[['TITLE','CATEGORY']]

dist = news['CATEGORY'].value_counts()
print(dist)
print(news.head())



# --------------
stop = set(stopwords.words('english'))
news['TITLE'] = news['TITLE'].apply(lambda x: re.sub("[^a-zA-Z]", " ",x))
news['TITLE'] = news['TITLE'].apply(lambda x: x.lower().split())

news['TITLE'] = news['TITLE'].apply(lambda x: [i for i in x if i not in stop])
news['TITLE'] = news['TITLE'].apply(lambda x: ' '.join(x))

X = news['TITLE']
y = news['CATEGORY']

X_train, X_test, Y_train ,Y_test = train_test_split(X,y,test_size=0.2,random_state = 3)


# --------------
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))

X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test) 

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test) 





# --------------
nb_1 = MultinomialNB()
nb_2 = MultinomialNB()

nb_1.fit(X_train_count,Y_train)
nb_2.fit(X_train_tfidf,Y_train)

acc_count_nb = accuracy_score(nb_1.predict(X_test_count), Y_test)
acc_tfidf_nb = accuracy_score(nb_2.predict(X_test_tfidf), Y_test)

print(acc_count_nb)
print(acc_tfidf_nb)



# --------------
import warnings
warnings.filterwarnings('ignore')

logreg_1 = OneVsRestClassifier(LogisticRegression(random_state=10))
logreg_2 = OneVsRestClassifier(LogisticRegression(random_state=10))

logreg_1.fit(X_train_count,Y_train)
logreg_2.fit(X_train_tfidf,Y_train)

acc_count_logreg = accuracy_score(logreg_1.predict(X_test_count), Y_test)
acc_tfidf_logreg = accuracy_score(logreg_2.predict(X_test_tfidf), Y_test)

print(acc_count_logreg)
print(acc_tfidf_logreg)


