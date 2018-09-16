

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skew
from scipy import stats
import numpy as np

raw_train = pd.read_csv('train.csv')    # ID is the index
raw_test = pd.read_csv('test.csv')

print(raw_train.head())

print (raw_train.columns)
print (raw_train.shape)

############################### Train ####################################
# 1. Explore the correlation

# Top 10 heat map
k = 10
corrmat = raw_train.corr()
top = corrmat.nlargest(k, 'SalePrice').index  # Find the index of top 10 highest correlated variables
top_mat = corrmat.loc[top, top]               # Location their postions in heat map by index
fig, ax = plt.subplots(figsize=(8, 6))
sns.set(font_scale=1.25)
sns.heatmap(top_mat, annot=True, annot_kws={'size': 12}, square=True)
plt.show()

# 2. Check the na rate
def get_na_rate(dataframe):
    na_count = dataframe.isnull().sum()
    na_rate = na_count / len(dataframe)
    na_df = pd.concat([na_count, na_rate], axis=1, keys=['count', 'percent'])
    na_df = na_df.sort_values(['percent'], ascending=False)
    return na_df

df_na = get_na_rate(raw_train)
df_na.head(20)

# If na rate is higher than 0.5, then drop the column
drop_names = list(df_na[df_na.percent > 0.5].index) 
train = raw_train.drop(drop_names, axis=1)

# FireplaceQu's na rate is about 50% but we already have another feature "Fireplaces"
# Therefre we delete it
train = train.drop(['FireplaceQu'],axis=1)

# LotFrontage's na rate is about 20%, we fill the na with mean.
train['LotFrontage'] = train['LotFrontage'].fillna(train['LotFrontage'].mean())

# If na rate is lower then 1%, we consider deleting them
# Since there are two types of variables with na rate lower than 1%,
# one is related to garage, another is related to basement
# Therefore, we create two new variables to indicate the presence of 
# garage and basement
train['hasGarage'] = train['GarageYrBlt'].notnull().astype(int)
train['hasBasement'] = train['BsmtQual'].notnull().astype(int)

train = train.dropna(axis = 'columns')
get_na_rate(train)

# 3. Explore the Saleprice

train['SalePrice'].describe()

sns.distplot(train[['SalePrice']])
fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)

# Normalize the Saleprice
train['SalePrice'] = np.log(train['SalePrice'])
sns.distplot(train[['SalePrice']])
fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)

# 4. Classify the data to the qualitative and quantitative data

train['MSSubClass'] = train['MSSubClass'].astype(str)        # Convert MSSubClass to categorical data
qualitative = [qual for qual in train.columns if train.dtypes[qual] == 'O']
quantitative = [quan for quan in test.columns if quan not in qualitative]
print("qualitative: {}, quantitative: {}" .format (len(qualitative),len(quantitative)))

# Have a look at the skewness of quantitative data
melt = pd.melt(train, value_vars=quantitative)
splot = sns.FacetGrid(melt, col="variable",  col_wrap=4, sharex=False, sharey=False)
splot.map(sns.distplot, "value")

df_skew = pd.DataFrame({'Variable':quantitative, 'Skewness':skew(train[quantitative])})
df_skew = df_skew.sort_values(by=['Skewness'], ascending=False)
df_skew

# Normalize some quantitative data by log transformation
log_cols = ['LotFrontage','LotArea','1stFlrSF','GrLivArea','KitchenAbvGr']

for col in log_cols:
    train[col] = np.log1p(train[col].values)

df_skew_new = pd.DataFrame({'Variable':quantitative, 'Skewness':skew(train[quantitative])})
df_skew_new = df_skew_new.sort_values(by=['Skewness'], ascending=False)
df_skew_new

melt = pd.melt(train, value_vars=log_cols)
splot = sns.FacetGrid(melt, col="variable",  col_wrap=4, sharex=False, sharey=False)
splot.map(sns.distplot, "value")

# Check the amount of zero
def get_zero_rate(dataframe):
    zero_count = len(dataframe) - dataframe.astype(bool).sum(axis=0)
    zero_rate = zero_count / len(dataframe)
    zero_df = pd.concat([zero_count, zero_rate], axis=1, keys=['count', 'percent'])
    zero_df = zero_df.sort_values(['percent'],ascending=False)
    return zero_df

df_zero = get_zero_rate(train).head(10)
df_zero

# If one column has too many zero ( > 90%), then delete the columns
drop_cols = list(df_zero[df_zero.percent > 0.9].index) 
train = train.drop(drop_cols, axis=1)

# For those columns with percentage of zero about 50%, convert them to the dummy variable
def convert_to_boolean(column):
    return train[column].apply(lambda x: 0 if x > 0 else 1)

train['has2ndFlr'] = convert_to_boolean('2ndFlrSF')
train['hasWoodDeck'] = convert_to_boolean('WoodDeckSF')
train['hasOpenPorch'] = convert_to_boolean('OpenPorchSF')

train.shape
# train.to_csv('train_after_cleaning.csv',index=False)

############################### Test ####################################

test = raw_test.drop(drop_names, axis=1)

test = test.drop(['FireplaceQu'],axis=1)

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())

test['hasGarage'] = test['GarageYrBlt'].notnull().astype(int)
test['hasBasement'] = test['BsmtQual'].notnull().astype(int)

test = test.dropna(axis = 'columns')
get_na_rate(test)

test['MSSubClass'] = train['MSSubClass'].astype(str)        # Convert MSSubClass to categorical data
qualitative = [qual for qual in test.columns if test.dtypes[qual] == 'O']
quantitative = [quan for quan in test.columns if quan not in qualitative]
print("qualitative: {}, quantitative: {}" .format (len(qualitative),len(quantitative)))

# Have a look at the skewness of quantitative data
melt = pd.melt(train, value_vars=quantitative)
splot = sns.FacetGrid(melt, col="variable",  col_wrap=4, sharex=False, sharey=False)
splot.map(sns.distplot, "value")

df_skew = pd.DataFrame({'Variable':quantitative, 'Skewness':skew(train[quantitative])})
df_skew = df_skew.sort_values(by=['Skewness'], ascending=False)
df_skew

# Normalize some quantitative data by log transformation
log_cols = ['LotFrontage','LotArea','1stFlrSF','GrLivArea','KitchenAbvGr']

for col in log_cols:
    train[col] = np.log1p(train[col].values)

df_skew_new = pd.DataFrame({'Variable':quantitative, 'Skewness':skew(train[quantitative])})
df_skew_new = df_skew_new.sort_values(by=['Skewness'], ascending=False)
df_skew_new

melt = pd.melt(train, value_vars=log_cols)
splot = sns.FacetGrid(melt, col="variable",  col_wrap=4, sharex=False, sharey=False)
splot.map(sns.distplot, "value")

# Check the amount of zero
def get_zero_rate(dataframe):
    zero_count = len(dataframe) - dataframe.astype(bool).sum(axis=0)
    zero_rate = zero_count / len(dataframe)
    zero_df = pd.concat([zero_count, zero_rate], axis=1, keys=['count', 'percent'])
    zero_df = zero_df.sort_values(['percent'],ascending=False)
    return zero_df

df_zero = get_zero_rate(train).head(10)
df_zero

# If one column has too many zero ( > 90%), then delete the columns
drop_cols = list(df_zero[df_zero.percent > 0.9].index) 
train = train.drop(drop_cols, axis=1)

# For those columns with percentage of zero about 50%, convert them to the dummy variable
def convert_to_boolean(column):
    return train[column].apply(lambda x: 0 if x > 0 else 1)

train['has2ndFlr'] = convert_to_boolean('2ndFlrSF')
train['hasWoodDeck'] = convert_to_boolean('WoodDeckSF')
train['hasOpenPorch'] = convert_to_boolean('OpenPorchSF')

train.shape