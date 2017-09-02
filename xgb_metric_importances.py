import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

dpath = './data/'
data = pd.read_csv(dpath+"mushrooms.csv")
labelencoder=LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])
X = data.iloc[:,1:23]  # all rows, all the features and no labels
y = data.iloc[:, 0]  # all rows, label only
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)

model_XGB=XGBClassifier()
model_XGB.fit(X_train,y_train)
y_prob = model_XGB.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
score=model_XGB.score(X_test, y_test)
print(score)
auc_roc=metrics.roc_auc_score(y_test,y_pred)
print(auc_roc)
precision, recall, threshold = metrics.precision_recall_curve(y_test,y_pred)
print(precision)
print(recall)
print(threshold)
confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)


print(model_XGB.feature_importances_)
pyplot.bar(range(len(model_XGB.feature_importances_)), model_XGB.feature_importances_)
pyplot.show()
plot_importance(model_XGB)
pyplot.show()

# Fit model using each importance as a threshold
thresholds = np.sort(model_XGB.feature_importances_)
for thresh in thresholds:
  # select features using threshold
  selection = SelectFromModel(model_XGB, threshold=thresh, prefit=True)
  select_X_train = selection.transform(X_train)
  # train model
  selection_model = XGBClassifier()
  selection_model.fit(select_X_train, y_train)
# eval model
  select_X_test = selection.transform(X_test)
  y_pred = selection_model.predict(select_X_test)
  predictions = [round(value) for value in y_pred]
  accuracy = accuracy_score(y_test, predictions)
  print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],
      accuracy*100.0))