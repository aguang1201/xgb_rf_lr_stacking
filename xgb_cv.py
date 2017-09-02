import numpy as np
import xgboost as xgb
from matplotlib import pyplot

test_size = 550000
# path to where the data lies
dpath = './data'
graph = './graph/'
model = './model/'
out = './out/'
model_file = model + 'higgs_cv.model'
graph_file = graph+'HiggsBoson_estimators.png'
outfile = out+'higgs.pred.csv'
# load in training data, directly use numpy
dtrain = np.loadtxt(dpath + '/higgsboson_training.csv', delimiter=',', skiprows=1,
                converters={32: lambda x: int(x == 's'.encode('utf-8'))})
print ('finish loading from csv ')
label  = dtrain[:,32]
data   = dtrain[:,1:31]
# rescale weight to make it same as test set
weight = dtrain[:,31] * float(test_size) / len(label)
sum_wpos = sum( weight[i] for i in range(len(label)) if label[i] == 1.0  )
sum_wneg = sum( weight[i] for i in range(len(label)) if label[i] == 0.0  )
# print weight statstics
print ('weight statistics: wpos=%g, wneg=%g, ratio=%g' % ( sum_wpos, sum_wneg, sum_wneg/sum_wpos ))

dtrain = xgb.DMatrix( data, label=label, missing = -999.0, weight=weight )
# setup parameters for xgboost
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = -1
param['tree_method'] = 'gpu_hist'
param['n_gpus'] = -1

num_round = 1000
print ('running cross validation, with preprocessing function')
def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    wtrain = dtrain.get_weight()
    wtest = dtest.get_weight()
    sum_weight = sum(wtrain) + sum(wtest)
    wtrain *= sum_weight / sum(wtrain)
    wtest *= sum_weight / sum(wtest)
    dtrain.set_weight(wtrain)
    dtest.set_weight(wtest)
    return (dtrain, dtest, param)
cvresult = xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'ams@0.15', 'auc'}, early_stopping_rounds=10, seed = 0, fpreproc = fpreproc)
print ('finish cross validation')
n_estimators = cvresult.shape[0]
print ('n_estimators is:',n_estimators)
test_means = cvresult['test-ams@0.15-mean']
test_stds = cvresult['test-ams@0.15-std']
train_means = cvresult['train-ams@0.15-mean']
train_stds = cvresult['train-ams@0.15-std']
x_axis = range(0, n_estimators)
pyplot.errorbar(x_axis, test_means, yerr=test_stds, label='Test')
pyplot.errorbar(x_axis, train_means, yerr=train_stds, label='Train')
pyplot.title("HiggsBoson n_estimators vs ams@0.15")
pyplot.xlabel('n_estimators')
pyplot.ylabel('ams@0.15')
pyplot.savefig(graph_file)
pyplot.show()

print ('train model using the best parameters by cv ... ')
bst = xgb.train(param, dtrain, num_boost_round=n_estimators)
bst.save_model(model_file)
print ('retrain finished')

threshold_ratio = 0.15
dtest = np.loadtxt( dpath+'/higgsboson_test.csv', delimiter=',', skiprows=1 )
data   = dtest[:,1:31]
idx = dtest[:,0]
print ('finish loading from csv ')

xgmat = xgb.DMatrix( data, missing = -999.0 )
bst = xgb.Booster({'nthread':8}, model_file = model_file)
ypred = bst.predict( xgmat )

res  = [ ( int(idx[i]), ypred[i] ) for i in range(len(ypred)) ]

rorder = {}
for k, v in sorted( res, key = lambda x:-x[1] ):
    rorder[ k ] = len(rorder) + 1

# write out predictions
ntop = int( threshold_ratio * len(rorder ) )
fo = open(outfile, 'w')
nhit = 0
ntot = 0
fo.write('EventId,RankOrder,Class\n')
for k, v in res:
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'
    # change output rank order to follow Kaggle convention
    fo.write('%s,%d,%s\n' % ( k,  len(rorder)+1-rorder[k], lb ) )
    ntot += 1
fo.close()

print ('finished writing into prediction file')