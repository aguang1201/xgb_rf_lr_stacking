import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from MeanEncoder import MeanEncoder
from  scipy.io import mmwrite

dpath = './data/'
out = './out/'
train = pd.read_json(dpath +"RentListingInquries_train.json")
test = pd.read_json(dpath+"RentListingInquries_test.json")
y_map = {'low': 2, 'medium': 1, 'high': 0}
train['interest_level'] = train['interest_level'].apply(lambda x: y_map[x])
y_train = train.interest_level
train = train.drop(['listing_id', 'interest_level'], axis=1)
listing_id = test.listing_id.values
test = test.drop('listing_id', axis=1)
ntrain = train.shape[0]
train_test = pd.concat((train, test), axis=0).reset_index(drop=True)
train_test['price'].ix[train_test['price']>13000] = 13000
train_test.loc[train_test["bathrooms"] == 112, "bathrooms"] = 1.5
train_test.loc[train_test["bathrooms"] == 10, "bathrooms"] = 1
train_test.loc[train_test["bathrooms"] == 20, "bathrooms"] = 2
train_test['price_bathrooms'] =  (train_test["price"])/ (train_test["bathrooms"] +1.0)
train_test['price_bedrooms'] =  (train_test["price"])/ (train_test["bedrooms"] +1.0)
train_test["room_diff"] = train_test["bathrooms"] - train_test["bedrooms"]
train_test["room_num"] = train_test["bedrooms"] + train_test["bathrooms"]
train_test['Date'] = pd.to_datetime(train_test['created'])
train_test['Year'] = train_test['Date'].dt.year
train_test['Month'] = train_test['Date'].dt.month
train_test['Day'] = train_test['Date'].dt.day
train_test['Wday'] = train_test['Date'].dt.dayofweek
train_test['Yday'] = train_test['Date'].dt.dayofyear
train_test['hour'] = train_test['Date'].dt.hour
train_test = train_test.drop(['Date', 'created'], axis=1)
train_test["num_description_words"] = train_test["description"].apply(lambda x: len(x.split(" ")))
train_test = train_test.drop(['description'], axis=1)

managers_count = train_test['manager_id'].value_counts()
train_test['top_10_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 90)] else 0)
train_test['top_25_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 75)] else 0)
train_test['top_5_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 95)] else 0)
train_test['top_50_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 50)] else 0)
train_test['top_1_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 99)] else 0)
train_test['top_2_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 98)] else 0)
train_test['top_15_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 85)] else 0)
train_test['top_20_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 80)] else 0)
train_test['top_30_manager'] = train_test['manager_id'].apply(lambda x: 1 if x in managers_count.index.values[
    managers_count.values >= np.percentile(managers_count.values, 70)] else 0)

buildings_count = train_test['building_id'].value_counts()
train_test['top_10_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 90)] else 0)
train_test['top_25_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 75)] else 0)
train_test['top_5_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 95)] else 0)
train_test['top_50_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 50)] else 0)
train_test['top_1_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 99)] else 0)
train_test['top_2_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 98)] else 0)
train_test['top_15_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 85)] else 0)
train_test['top_20_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 80)] else 0)
train_test['top_30_building'] = train_test['building_id'].apply(lambda x: 1 if x in buildings_count.index.values[
    buildings_count.values >= np.percentile(buildings_count.values, 70)] else 0)

train_test['photos_count'] = train_test['photos'].apply(lambda x: len(x))
train_test.drop(['photos'], axis=1, inplace=True)

train_location = train_test.loc[:ntrain-1, ['latitude', 'longitude']]
test_location = train_test.loc[ntrain:, ['latitude', 'longitude']]
kmeans_cluster = KMeans(n_clusters=20)
res = kmeans_cluster.fit(train_location)
res = kmeans_cluster.predict( pd.concat((train_location, test_location), axis=0).reset_index(drop=True))
train_test['cenroid'] = res
# L1 distance
center = [ train_location['latitude'].mean(), train_location['longitude'].mean()]
train_test['distance'] = abs(train_test['latitude'] - center[0]) + abs(train_test['longitude'] - center[1])
train_test['display_address'] = train_test['display_address'].apply(lambda x: x.lower().strip())
train_test['street_address'] = train_test['street_address'].apply(lambda x: x.lower().strip())

categoricals = ['building_id', 'manager_id', 'display_address', 'street_address']
for feat in categoricals:
    lbl = LabelEncoder()
    lbl.fit(list(train_test[feat].values))
    train_test[feat] = lbl.transform(list(train_test[feat].values))
me = MeanEncoder(categoricals)
train_new = train_test.iloc[:ntrain, :]
train_new_cat = me.fit_transform(train_new, y_train)
#test
test_new = train_test.iloc[ntrain:, :]
test_new_cat = me.transform(test_new)
train_test = pd.concat((train_new_cat, test_new_cat), axis=0).reset_index(drop=True)
train_test.drop(categoricals, axis=1, inplace=True)

train_test['features_count'] = train_test['features'].apply(lambda x: len(x))
train_test['features2'] = train_test['features']
train_test['features2'] = train_test['features2'].apply(lambda x: ' '.join(x))
c_vect = CountVectorizer(stop_words='english', max_features=200, ngram_range=(1, 1))
c_vect_sparse = c_vect.fit_transform(train_test['features2'])
c_vect_sparse_cols = c_vect.get_feature_names()
train_test.drop(['features', 'features2'], axis=1, inplace=True)
train_test_sparse = sparse.hstack([train_test, c_vect_sparse]).tocsr()

train_test_new = pd.DataFrame(train_test_sparse.toarray())
X_train = train_test_new.iloc[:ntrain, :]
X_test = train_test_new.iloc[ntrain:, :]
train_new = pd.concat((X_train, y_train), axis=1).reset_index(drop=True)
train_new.to_csv(out + 'RentListingInquries_FE_train.csv', index=False)
X_test.to_csv(out + 'RentListingInquries_FE_test.csv', index=False)

X_train_sparse = train_test_sparse[:ntrain, :]
X_test_sparse = train_test_sparse[ntrain:, :]
train_sparse = sparse.hstack([X_train_sparse, sparse.csr_matrix(y_train).T]).tocsr()
mmwrite(out + 'RentListingInquries_FE_train.txt',train_sparse)
mmwrite(out + 'RentListingInquries_FE_test.txt',X_test_sparse)