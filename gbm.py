#!/usr/bin/python
# this is the example script to use xgboost to train
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import time


test_size = 550000

# path to where the data lies
dpath = './data/'

# load in training data, directly use numpy
dtrain = np.loadtxt( dpath+'/higgsboson_training.csv', delimiter=',', skiprows=1, converters={32: lambda x:int(x=='s') } )
print ('finish loading from csv ')
label  = dtrain[:,32]
data   = dtrain[:,1:31]
num_round = 10
print ('loading data end, start to boost trees')
print ("training GBM from sklearn")
tmp = time.time()
gbm = GradientBoostingClassifier(n_estimators=num_round, max_depth=6, verbose=2)
gbm.fit(data, label)
print ("sklearn.GBM costs: %s seconds" % str(time.time() - tmp))