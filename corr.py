import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

dpath = './data/'
graph = './graph/'
train_rent = pd.read_json(dpath + "RentListingInquries_train.json")
contFeatureslist = []
contFeatureslist.append("bathrooms")
contFeatureslist.append("bedrooms")
contFeatureslist.append("price")

correlationMatrix = train_rent[contFeatureslist].corr().abs()
plt.subplots(figsize=(13, 9))
sns.heatmap(correlationMatrix,annot=True)
sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)
graph_file = graph+'corr_heatmap_bathrooms_bedrooms_price.png'
pyplot.savefig(graph_file)

train = pd.read_csv(dpath + "AllstateClaimsSeverity_train.csv")
split = 117
size = 15
data=train.iloc[:,split:]
cols=data.columns
data_corr = data.corr().abs()
plt.subplots(figsize=(13, 9))
sns.heatmap(data_corr,annot=True)
sns.heatmap(data_corr, mask=data_corr < 1, cbar=False)
graph_file = graph+'corr_heatmap_all.png'
pyplot.savefig(graph_file)

threshold = 0.5
corr_list = []
#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index
#Sort to show higher ones first
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))
for v,i,j in s_corr_list:
    fig = plt.figure()
    sns.pairplot(train, size=6, x_vars=cols[i],y_vars=cols[j] )
    graph_file = graph + 'corr_pairplot_'+cols[i]+'_'+cols[j]+'.png'
    pyplot.savefig(graph_file)