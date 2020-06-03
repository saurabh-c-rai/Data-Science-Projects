# --------------
import pandas as pd
from collections import Counter

# Load dataset
data = pd.read_csv(path)
print(data.isnull().sum())
print(data.info())





# --------------
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style(style='darkgrid')

# Store the label values 
label = data['Activity']
sns.countplot(label)
plt.xticks(rotation=90)
# plot the countplot



# --------------
data_copy = data.copy()

data_copy['duration'] = ''
#label[(label=='WALKING_UPSTAIRS') | (label=='WALKING_DOWNSTAIRS')]

duration_df = (data_copy.groupby([label[label.isin(['WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'])], 'subject'])['duration'].count() * 1.28)
duration_df = pd.DataFrame(duration_df)

plot_data = duration_df.reset_index().sort_values('duration',ascending='False')
plot_data['Activity'] = plot_data['Activity'].map({'WALKING_UPSTAIRS':'Upstairs','WALKING_DOWNSTAIRS':'Downstairs'})

plt.figure(figsize=(15,6))
sns.barplot(data=plot_data, x='subject', y='duration', hue='Activity')
plt.show()


# --------------
feature_cols = data.select_dtypes('float').columns
correlated_values = data[feature_cols].corr()

correlated_values = correlated_values.stack().to_frame().reset_index().rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0:'Correlation_score'})

correlated_values['abs_correlation'] = abs(correlated_values['Correlation_score'])

s_corr_list = correlated_values.sort_values('abs_correlation',ascending = False)

top_corr_fields = s_corr_list[(s_corr_list['abs_correlation'] > 0.8) & (s_corr_list['Feature_1'] != s_corr_list['Feature_2'])]

print(top_corr_fields.shape)






# --------------
# importing neccessary libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import confusion_matrix, accuracy_score

# Encoding the target variable
le = LabelEncoder()
data['Activity']  =  le.fit_transform(data['Activity'])
X = data.drop('Activity',1)
y = data['Activity']

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 40)

# Baseline model 
classifier = SVC()
clf = classifier.fit(X_train,y_train)
y_pred = clf.predict(X_test)

precision,recall,f_score,support = error_metric(y_test,y_pred, average = 'weighted')

model1_score = accuracy_score(y_test,y_pred)

print('model1_score',model1_score)
print('precision',precision)
print('recall',recall)
print('f1_score',f_score)


# --------------
# importing libraries
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=0.01,penalty='l1',dual=False,random_state=42).fit(X_train,y_train)

model_2 = SelectFromModel(lsvc,prefit=True)
new_train_features = model_2.transform(X_train)
new_test_features = model_2.transform(X_test)

classfier_2 = SVC()
clf_2 = classfier_2.fit(new_train_features,y_train)
y_pred_new = clf_2.predict(new_test_features)

model2_score = accuracy_score(y_test,y_pred_new)
precision,recall,f_score,support = error_metric(y_test,y_pred_new, average = 'weighted')

print('model1_score',model2_score)
print('precision',precision)
print('recall',recall)
print('f1_score',f_score)



# --------------
# Importing Libraries
from sklearn.model_selection import GridSearchCV

# Set the hyperparmeters
parameters = {'kernel': ['linear', 'rbf'],'C': [100, 20, 1, 0.1]}
selector = GridSearchCV(SVC(),param_grid=parameters,scoring = 'accuracy')
selector.fit(new_train_features,y_train)
print(selector.best_params_)
means = selector.cv_results_['mean_test_score']
stds = selector.cv_results_['std_test_score']
params = selector.cv_results_['params']


classifier_3  = SVC(C= 100, kernel= 'rbf')
clf_3 = classifier_3.fit(new_train_features,y_train)
y_pred_final = clf_3.predict(new_test_features)

model3_score = accuracy_score(y_test,y_pred_final)
precision,recall,f_score,support = error_metric(y_test,y_pred_final, average = 'weighted')

print('model1_score',model3_score)
print('precision',precision)
print('recall',recall)
print('f1_score',f_score)




