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

max_depth = range(3,10,2)
min_child_weight = range(1,6,2)
param_test2_1 = dict(max_depth=max_depth, min_child_weight=min_child_weight)
xgb2_1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=671,
        max_depth=5,
        min_child_weight=1,
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
gsearch2_1 = GridSearchCV(xgb2_1, param_grid = param_test2_1, scoring='neg_log_loss',n_jobs=8, cv=kfold)
gsearch2_1.fit(X_train , y_train)
print gsearch2_1.grid_scores_
print gsearch2_1.best_params_
print gsearch2_1.best_score_

print("Best: %f using %s" % (gsearch2_1.best_score_, gsearch2_1.best_params_))
test_means = gsearch2_1.cv_results_['mean_test_score']
test_stds = gsearch2_1.cv_results_['std_test_score']
train_means = gsearch2_1.cv_results_['mean_train_score']
train_stds = gsearch2_1.cv_results_['std_train_score']
pd.DataFrame(gsearch2_1.cv_results_).to_csv('out/my_preds_maxdepth_min_child_weights_1.csv')
# plot results
test_scores = np.array(test_means).reshape(len(max_depth), len(min_child_weight))
train_scores = np.array(train_means).reshape(len(max_depth), len(min_child_weight))
for i, value in enumerate(max_depth):
    pyplot.plot(min_child_weight, -test_scores[i], label='test_max_depth:' + str(value))
pyplot.legend()
pyplot.xlabel('max_depth')
pyplot.ylabel('Log Loss')
pyplot.savefig('graph/max_depth_vs_min_child_weght_1.png')