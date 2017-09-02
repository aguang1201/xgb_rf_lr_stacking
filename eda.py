import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from wordcloud import WordCloud

dpath = './data/'
graph = './graph/'
train = pd.read_json(dpath +"RentListingInquries_train.json")
test = pd.read_json(dpath+"RentListingInquries_test.json")
order = ['low', 'medium', 'high']

sns.countplot(train.interest_level, order=order)
plt.xlabel('Interest Level')
plt.ylabel('Number of occurrences')
graph_file = graph+'countplot_interest_level.png'
pyplot.savefig(graph_file)

# train['interest'] = np.where(train.interest_level=='low',0,np.where(train.interest_level=='medium',1,2))
target_num_map = {'high':2, 'medium':1, 'low':0}
y = train["interest_level"].apply(lambda x: target_num_map[x])
#------------------------------------------------bathrooms start-------------------------------------------------
fig = plt.figure()
sns.countplot(train.bathrooms)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Number of occurrences')
graph_file = graph+'countplot_bathrooms.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["bathrooms"],jitter=True,order=order)
plt.title("Number of Number of Bathrooms Vs Interest_level")
graph_file = graph+'stripplot_bathrooms.png'
pyplot.savefig(graph_file)

ulimit = 4
train['bathrooms'].ix[train['bathrooms']>ulimit] = ulimit

fig = plt.figure()
sns.countplot(train.bathrooms)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Number of occurrences')
graph_file = graph+'countplot_bathrooms_fixed.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["bathrooms"],jitter=True,order=order)
plt.title("Number of Number of Bathrooms Vs Interest_level")
graph_file = graph+'stripplot_bathrooms_fixed.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.countplot(x="bathrooms", hue="interest_level",data=train)
graph_file = graph+'countplot_hue_bathrooms_fixed.png'
pyplot.savefig(graph_file)
#------------------------------------------------bathrooms end-------------------------------------------------
#------------------------------------------------bedrooms start-------------------------------------------------
fig = plt.figure()
sns.countplot(train.bedrooms)
plt.xlabel('Number of Bedrooms')
plt.ylabel('Number of occurrences')
graph_file = graph+'countplot_bedrooms.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["bedrooms"],jitter=True,order=order)
plt.title("Number of Bedrooms Vs Interest_level")
graph_file = graph+'stripplot_bedrooms.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.countplot(x="bedrooms", hue="interest_level",data=train)
graph_file = graph+'countplot_hue_bedrooms.png'
pyplot.savefig(graph_file)
#------------------------------------------------bedrooms end-------------------------------------------------
#------------------------------------------------price start-------------------------------------------------
fig = plt.figure()
plt.scatter(range(train.shape[0]), train["price"].values,color='purple')
plt.title("Distribution of Price")
graph_file = graph+'scatter_price.png'
pyplot.savefig(graph_file)

ulimit = np.percentile(train.price.values, 99)
train['price'].ix[train['price']>ulimit] = ulimit
fig = plt.figure()
sns.distplot(train.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
graph_file = graph+'distplot_price_fixed.png'
pyplot.savefig(graph_file)

plt.figure(figsize=(13,9))
sns.distplot(np.log1p(train["price"]))
graph_file = graph+'distplot_price_fixed_log.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["price"],jitter=True,order=order)
plt.title("Price Vs Interest_level")
graph_file = graph+'stripplot_price_fixed.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.violinplot(x='interest_level', y='price', data=train, order = order)
plt.xlabel('Interest level', fontsize=12)
plt.ylabel('price', fontsize=12)
graph_file = graph+'violinplot_price_fixed.png'
pyplot.savefig(graph_file)
#------------------------------------------------price end-------------------------------------------------
#------------------------------------------------listing_id start-------------------------------------------------
fig = plt.figure()
sns.distplot(train.listing_id.values, bins=50, kde=True)
plt.xlabel('listing_id')
graph_file = graph+'distplot_listingid.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["listing_id"],jitter=True,order=order)
plt.title("listing_id Vs Interest_level")
graph_file = graph+'stripplot_listingid.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.violinplot(x='interest_level', y='listing_id', data=train, order = order)
plt.xlabel('Interest level', fontsize=12)
plt.ylabel('price', fontsize=12)
graph_file = graph+'violinplot_listingid.png'
pyplot.savefig(graph_file)
#------------------------------------------------listing_id end-------------------------------------------------
#------------------------------------------------Longitude Latitude start-------------------------------------------------
sns.lmplot(x="longitude", y="latitude", fit_reg=False, hue='interest_level',
           hue_order=order, size=9, scatter_kws={'alpha':0.4,'s':30},
           data=train[(train.longitude>train.longitude.quantile(0.005))
                           &(train.longitude<train.longitude.quantile(0.995))
                           &(train.latitude>train.latitude.quantile(0.005))
                           &(train.latitude<train.latitude.quantile(0.995))])
plt.xlabel('Longitude')
plt.ylabel('Latitude')
graph_file = graph+'lmplot_longitude_latitude.png'
pyplot.savefig(graph_file)
#------------------------------------------------Longitude Latitude end-------------------------------------------------
#------------------------------------------------display_address start-------------------------------------------------
cnt_srs = train.groupby('display_address')['display_address'].count()
for i in [2, 10, 50, 100, 500]:
    print('Display_address that appear less than {} times: {}%'.format(i, round((cnt_srs < i).mean() * 100, 2)))
plt.figure()
plt.hist(cnt_srs.values, bins=100, log=True, alpha=0.9)
plt.xlabel('Number of times display_address appeared')
plt.ylabel('log(Count)')
graph_file = graph+'hist_display_address.png'
pyplot.savefig(graph_file)

### Let's get a list of top 10 display address
top10da = train.display_address.value_counts().nlargest(10).index.tolist()
fig = plt.figure()
ax = sns.countplot(x="display_address", hue="interest_level",
                   data=train[train.display_address.isin(top10da)])
plt.xlabel('display_address');
plt.ylabel('Number of advert occurrences');
### Manager_ids are too long. Let's remove them
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')
### Adding percents over bars
height = [0 if np.isnan(p.get_height()) else p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 20,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center")
graph_file = graph+'countplot_display_address.png'
pyplot.savefig(graph_file)
#------------------------------------------------display_address end-------------------------------------------------
#------------------------------------------------building_id start-------------------------------------------------
### Let's get a list of top 10 building id
top10building = train.building_id.value_counts().nlargest(10).index.tolist()
### ...and plot number of different Interest Level rental adverts for each of them
fig = plt.figure()
ax = sns.countplot(x="building_id", hue="interest_level",
                   data=train[train.building_id.isin(top10building)]);
plt.xlabel('Biulding');
plt.ylabel('Number of advert occurrences');
### Manager_ids are too long. Let's remove them
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off');

### Adding percents over bars
height = [0 if np.isnan(p.get_height()) else p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 20,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center")
graph_file = graph+'countplot_building_id.png'
pyplot.savefig(graph_file)
#------------------------------------------------building_id end-------------------------------------------------
#------------------------------------------------manager_id start-------------------------------------------------
### Let's get a list of top 10 managers
top10managers = train.manager_id.value_counts().nlargest(10).index.tolist()
### ...and plot number of different Interest Level rental adverts for each of them
fig = plt.figure()
ax = sns.countplot(x="manager_id", hue="interest_level",
                   data=train[train.manager_id.isin(top10managers)]);
plt.xlabel('Manager');
plt.ylabel('Number of advert occurrences');
### Manager_ids are too long. Let's remove them
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off');

### Adding percents over bars
height = [0 if np.isnan(p.get_height()) else p.get_height() for p in ax.patches]
ncol = int(len(height)/3)
total = [height[i] + height[i + ncol] + height[i + 2*ncol] for i in range(ncol)] * 3
for i, p in enumerate(ax.patches):
    ax.text(p.get_x()+p.get_width()/2,
            height[i] + 20,
            '{:1.0%}'.format(height[i]/total[i]),
            ha="center")
graph_file = graph + 'countplot_manager_id.png'
pyplot.savefig(graph_file)
#------------------------------------------------manager_id end-------------------------------------------------
#------------------------------------------------created start-------------------------------------------------
train['created'] = pd.to_datetime(train['created'])
train['date'] = train['created'].dt.date
train["year"] = train["created"].dt.year
train['month'] = train['created'].dt.month
train['day'] = train['created'].dt.day
train['hour'] = train['created'].dt.hour
train['weekday'] = train['created'].dt.weekday
train['week'] = train['created'].dt.week
train['quarter'] = train['created'].dt.quarter
train['weekend'] = ((train['weekday'] == 5) & (train['weekday'] == 6))
train['wd'] = ((train['weekday'] != 5) & (train['weekday'] != 6))
cnt_srs = train['date'].value_counts()
plt.figure(figsize=(12,4))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values)
ax.xaxis_date()
plt.xticks(rotation='vertical')
graph_file = graph + 'bar_date_counts.png'
pyplot.savefig(graph_file)

fig = plt.figure()
hourDF = train.groupby(['hour', 'interest_level'])['hour'].count().unstack('interest_level').fillna(0)
hourDF[order].plot(kind='bar', stacked=True)
graph_file = graph + 'bar_hour_counts.png'
pyplot.savefig(graph_file)

fig = plt.figure()
monthDF = train.groupby(['month', 'interest_level'])['month'].count().unstack('interest_level').fillna(0)
monthDF[order].plot(kind='bar', stacked=True)
graph_file = graph + 'bar_month_counts.png'
pyplot.savefig(graph_file)
#------------------------------------------------created end-------------------------------------------------
#------------------------------------------------num_photos start-------------------------------------------------
train['num_photos'] = train['photos'].apply(len)
ulimit = np.percentile(train.num_photos.values, 99)
train['num_photos'].ix[train['num_photos']>ulimit] = ulimit
fig = plt.figure()
sns.countplot(train.num_photos)
plt.xlabel('Number of photoes')
plt.ylabel('Number of occurrences')
graph_file = graph + 'countplot_num_photos.png'
pyplot.savefig(graph_file)

train['num_photos'].ix[train['num_photos']>15] = 15
plt.figure()
sns.violinplot(x="num_photos", y="interest_level", data=train, order =order)
plt.xlabel('Number of Photos')
plt.ylabel('Interest Level')
graph_file = graph + 'violinplot_num_photos_fixed.png'
pyplot.savefig(graph_file)
#------------------------------------------------num_photos end-------------------------------------------------
#------------------------------------------------len_features start-------------------------------------------------
train['len_features'] = train['features'].apply(len)
plt.figure()
sns.countplot(train.len_features)
plt.xlabel('Length of features')
plt.ylabel('Number of occurrences')
graph_file = graph + 'countplot_len_features.png'
pyplot.savefig(graph_file)

train['len_features'].ix[train['len_features'] > 16] = 16
plt.figure()
sns.violinplot(x="len_features", y="interest_level", data=train, order =order)
plt.xlabel('Length of Features')
plt.ylabel('Interest Level')
graph_file = graph + 'countplot_len_features_fixed.png'
pyplot.savefig(graph_file)
#------------------------------------------------len_features end-------------------------------------------------
#------------------------------------------------description start-------------------------------------------------
train['num_description_words'] = train['description'].apply(lambda x: len(x.split(' ')))
train['len_description'] = train['description'].apply(len)
plt.figure()
sns.countplot(train.len_description)
plt.xlabel('Length of description')
plt.ylabel('Number of occurrences')
graph_file = graph + 'countplot_description.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["len_description"],jitter=True,order=order)
plt.title("Length of description Vs Interest_level")
graph_file = graph + 'stripplot_description.png'
pyplot.savefig(graph_file)

plt.figure()
sns.violinplot(x="len_description", y="interest_level", data=train, order =order)
plt.xlabel('Length of Description')
plt.ylabel('Interest Level')
graph_file = graph + 'violinplot_description.png'
pyplot.savefig(graph_file)

plt.figure()
sns.countplot(train.num_description_words)
plt.xlabel('Number of words of description')
plt.ylabel('Number of occurrences')
graph_file = graph + 'countplot_num_description_words.png'
pyplot.savefig(graph_file)

fig = plt.figure()
sns.stripplot(train["interest_level"],train["num_description_words"],jitter=True,order=order)
plt.title("Length of description Vs Interest_level")
graph_file = graph + 'stripplot_num_description_words.png'
pyplot.savefig(graph_file)

plt.figure()
sns.violinplot(x="num_description_words", y="interest_level", data=train, order =order)
plt.xlabel('Number of Description Words')
plt.ylabel('Interest Level')
graph_file = graph + 'violinplot_num_description_words.png'
pyplot.savefig(graph_file)



text = ''
text_da = ''
text_street = ''
#text_desc = ''
for ind, row in train.iterrows():
    for feature in row['features']:
        text = " ".join([text, "_".join(feature.strip().split(" "))])
    text_da = " ".join([text_da,"_".join(row['display_address'].strip().split(" "))])
    text_street = " ".join([text_street,"_".join(row['street_address'].strip().split(" "))])
    #text_desc = " ".join([text_desc, row['description']])
text = text.strip()
text_da = text_da.strip()
text_street = text_street.strip()
#text_desc = text_desc.strip()
plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for features", fontsize=30)
plt.axis("off")
graph_file = graph + 'wordcloud_feature.png'
pyplot.savefig(graph_file)

# wordcloud for display address
plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text_da)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for Display Address", fontsize=30)
plt.axis("off")
graph_file = graph + 'wordcloud_display_address.png'
pyplot.savefig(graph_file)

# wordcloud for street address
plt.figure(figsize=(12,6))
wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text_street)
wordcloud.recolor(random_state=0)
plt.imshow(wordcloud)
plt.title("Wordcloud for Street Address", fontsize=30)
plt.axis("off")
graph_file = graph + 'wordcloud_street_address.png'
pyplot.savefig(graph_file)
#------------------------------------------------description end-------------------------------------------------
