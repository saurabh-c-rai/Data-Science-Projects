# --------------
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv(path,sep ="\t")
print(df['date'].dtype)

df['date']= pd.to_datetime(df['date']) 
print(df['date'].dtype)

df['lenght'] = df['verified_reviews'].apply(lambda x: len(x))


# --------------
sns.countplot(x = 'rating', hue = 'feedback' , data = df)
plt.show()
sns.barplot(x = 'rating',y = 'variation' ,hue = 'feedback' , data = df)
plt.show()



# --------------
# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# declare empty list 'corpus'
corpus=[]

# for loop to fill in corpus
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]', ' ', df['verified_reviews'][i] )
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)


# --------------
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(corpus).toarray()

y = df['feedback']

count = y.value_counts()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# --------------
# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

rf = RandomForestClassifier(random_state=2)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

score = accuracy_score(y_test,y_pred)

precision = precision_score(y_test,y_pred)

print(score)
print(precision)


# --------------
# import packages
from imblearn.over_sampling import SMOTE

# Instantiate smote
smote = SMOTE(random_state=9)

# fit_sample onm training data
X_train, y_train = smote.fit_sample(X_train, y_train)

# fit modelk on training data
rf.fit(X_train, y_train)

# predict on test data
y_pred = rf.predict(X_test)

# calculate the accuracy score
score = accuracy_score(y_test, y_pred)

# calculate the precision
precision = precision_score(y_test, y_pred)

# display precision and score
print(score, precision)


