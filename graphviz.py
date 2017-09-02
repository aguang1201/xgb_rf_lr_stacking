import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
from matplotlib import pyplot
import graphviz

my_workpath = './data/'
dtrain = xgb.DMatrix(my_workpath + 'agaricus.txt.train')
dtest = xgb.DMatrix(my_workpath + 'agaricus.txt.test')
print dtrain.num_col()
print dtrain.num_row()
print dtest.num_row()
param = {'max_depth':2, 'eta':1, 'silent':0, 'objective':'binary:logistic' }
num_round = 2
starttime = time.clock()
bst = xgb.train(param, dtrain, num_round)
endtime = time.clock()
print (endtime - starttime)

train_preds = bst.predict(dtrain)
train_predictions = [round(value) for value in train_preds]
y_train = dtrain.get_label()
train_accuracy = accuracy_score(y_train, train_predictions)
print ("Train Accuary: %.2f%%" % (train_accuracy * 100.0))

preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
y_test = dtest.get_label()
test_accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))

# xgb.plot_tree(bst, num_trees=0, rankdir= 'LR' )
xgb.plot_tree(bst,num_trees=1, rankdir= 'LR' )
pyplot.show()

#xgb.to_graphviz(bst,num_trees=0)
#xgb.to_graphviz(bst,num_trees=1)