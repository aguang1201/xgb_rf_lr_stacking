from xgboost import XGBClassifier
import xgboost as xgb

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from matplotlib import pyplot
import seaborn as sns

dpath = './data/'
train = pd.read_csv(dpath +"Otto_train.csv")
y_train = train['target']
y_train = y_train.map(lambda s: s[6:])
y_train = y_train.map(lambda s: int(s)-1)
train = train.drop(["id", "target"], axis=1)
X_train = np.array(train)
# prepare cross validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)

param = {'objective': 'multi:softprob', # Specify multiclass classification
         'num_class': 9, # Number of possible output classes
         'tree_method': 'gpu_hist', # Use GPU accelerated algorithm
         'early_stopping_rounds':20,
         'save_period':1,
         #'gpu_id':1,
         'n_gpus': -1,
         'nthread':4,
         }

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_train, label=y_train)
gpu_res = {} # Store accuracy result
# Train model
# model = xgb.train(param, dtrain, 300, evals=[(dtest, 'test')], evals_result=gpu_res,)

cvresult = xgb.cv(param, dtrain, num_boost_round=300, folds =kfold,
                         metrics='mlogloss', early_stopping_rounds=20)

# model.save_model('0001.model')
# model.dump_model('dump.raw.txt','featmap.txt')

# model = xgb.booster({'nthread':4})
# model.load_model("model.bin")