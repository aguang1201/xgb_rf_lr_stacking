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

reg_alpha = [ 1.5, 2]
reg_lambda = [0.5, 1, 2]
param_test5_1 = dict(reg_alpha=reg_alpha, reg_lambda=reg_lambda)
xgb5_1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=645,
        max_depth=6,
        min_child_weight=4,
        gamma=0,
        subsample=0.7,
        colsample_bytree=0.6,
        colsample_bylevel = 0.7,
        objective= 'multi:softprob',
        tree_method='gpu_hist',
        n_gpus=-1,
        n_jobs=-1,
        silent=0,
        seed=3)
gsearch5_1 = GridSearchCV(xgb5_1, param_grid = param_test5_1, scoring='neg_log_loss',n_jobs=8, cv=kfold)
gsearch5_1.fit(X_train , y_train)
print gsearch5_1.grid_scores_
print gsearch5_1.best_params_
print gsearch5_1.best_score_

print("Best: %f using %s" % (gsearch5_1.best_score_, gsearch5_1.best_params_))
test_means = gsearch5_1.cv_results_['mean_test_score']
test_stds = gsearch5_1.cv_results_['std_test_score']
train_means = gsearch5_1.cv_results_['mean_train_score']
train_stds = gsearch5_1.cv_results_['std_train_score']
pd.DataFrame(gsearch5_1.cv_results_).to_csv('out/my_preds_reg_alpha_reg_lambda_1.csv')
test_scores = np.array(test_means).reshape(len(reg_alpha), len(reg_lambda))
train_scores = np.array(train_means).reshape(len(reg_alpha), len(reg_lambda))
for i, value in enumerate(reg_alpha):
        pyplot.plot(reg_lambda, -test_scores[i], label='reg_alpha:' + str(value))
pyplot.legend()
pyplot.xlabel('reg_alpha')
pyplot.ylabel('-Log Loss')
pyplot.savefig('graph/reg_alpha_vs_reg_lambda1.png')