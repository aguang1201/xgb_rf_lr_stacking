import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

dpath = './data/'
data = pd.read_csv(dpath+"mushrooms.csv")
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

model_RR=RandomForestClassifier()
model_RR.fit(X_train,y_train)
y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
score=model_RR.score(X_test, y_test)
print(score)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)
precision, recall, threshold = metrics.precision_recall_curve(y_test,y_pred)
print(precision)
print(recall)
print(threshold)
confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#tuned the parameters of Random Forest
model_RR=RandomForestClassifier()
tuned_parameters = {'min_samples_leaf': range(1,10,2), 'n_estimators' : range(1,10,2) }
RR = GridSearchCV(model_RR, tuned_parameters,cv=10,n_jobs=-1)
RR.fit(X_train,y_train)
y_prob = RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
score=RR.score(X_test, y_test)
print(score)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)
precision, recall, threshold = metrics.precision_recall_curve(y_test,y_pred)
print(precision)
print(recall)
print(threshold)
confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(RR.best_score_)
print(RR.best_params_)