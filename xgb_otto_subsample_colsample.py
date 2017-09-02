from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot

dpath = './data/'
train = pd.read_csv(dpath +"Otto_train.csv")
y_train = train['target']
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s)-1)
train = train.drop(["id", "target"], axis=1)
X_train = np.array(train)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

subsample = [i/10.0 for i in range(3,9)]
colsample_bytree = [i/10.0 for i in range(6,10)]
param_test3_1 = dict(subsample=subsample, colsample_bytree=colsample_bytree)
xgb3_1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=645,
        max_depth=6,
        min_child_weight=4,
        gamma=0,
        subsample=0.3,
        colsample_bytree=0.8,
        colsample_bylevel = 0.7,
        objective= 'multi:softprob',
        tree_method='gpu_hist',
        n_gpus=-1,
        n_jobs=-1,
        silent=0,
        seed=3)
gsearch3_1 = GridSearchCV(xgb3_1, param_grid = param_test3_1, scoring='neg_log_loss',n_jobs=8, cv=kfold)
gsearch3_1.fit(X_train , y_train)
print gsearch3_1.grid_scores_
print gsearch3_1.best_params_
print gsearch3_1.best_score_

print("Best: %f using %s" % (gsearch3_1.best_score_, gsearch3_1.best_params_))
test_means = gsearch3_1.cv_results_['mean_test_score']
test_stds = gsearch3_1.cv_results_['std_test_score']
train_means = gsearch3_1.cv_results_['mean_train_score']
train_stds = gsearch3_1.cv_results_['std_train_score']
pd.DataFrame(gsearch3_1.cv_results_).to_csv('out/my_preds_subsampleh_colsample_bytree_1.csv')
test_scores = np.array(test_means).reshape(len(colsample_bytree), len(subsample))
train_scores = np.array(train_means).reshape(len(colsample_bytree), len(subsample))

for i, value in enumerate(colsample_bytree):
        pyplot.plot(subsample, -test_scores[i], label='test_colsample_bytree:' + str(value))
pyplot.legend()
pyplot.xlabel('subsample')
pyplot.ylabel('Log Loss')
pyplot.savefig('graph/subsample_vs_colsample_bytree1.png')