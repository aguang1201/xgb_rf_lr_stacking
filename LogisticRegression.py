import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

dpath = './data/'
data = pd.read_csv(dpath+"mushrooms.csv")
print(data.head(6))
print(data.isnull().sum())
print(data['class'].unique())
print(data.dtypes)
print(data.shape)
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
model_LR= LogisticRegression()
model_LR.fit(X_train,y_train)
y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
score=model_LR.score(X_test, y_test)
print(score)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)
precision, recall, threshold = metrics.precision_recall_curve(y_test,y_pred)
print(precision)
print(recall)
print(threshold)
confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#gridsearch start
model_LR_1 = LogisticRegression()
tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'penalty':['l1','l2']}
LR = GridSearchCV(model_LR_1,tuned_parameters,n_jobs=-1,cv=10)
LR.fit(X_train,y_train)
best_params=LR.best_params_
best_scores=LR.best_score_
print(best_params)
print(best_scores)
y_prob = LR.predict(X_test)[:,1]
y_pred = np.where(y_prob>0.5,1,0)
score1=LR.score(X_test,y_test)
print(score1)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
precision, recall, threshold = metrics.precision_recall_curve(y_test,y_pred)
confusion_matrix=metrics.confusion_matrix(y_test, y_pred)

print(auc_roc)
print(precision)
print(recall)
print(threshold)
print(confusion_matrix)