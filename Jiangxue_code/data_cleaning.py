

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import skew
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder

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
    na_df = na_df[na_df['percent']>0]
    na_df = na_df.sort_values(['percent'], ascending=False)
    return na_df

df_na = get_na_rate(raw_train)
df_na.head(20)

# If na rate is higher than 0.5, then drop the column
drop_names = list(df_na[df_na.percent > 0.6].index) 
train = raw_train.drop(drop_names, axis=1)

# FireplaceQu's na rate is about 50% but we already have another feature "Fireplaces"
# Therefre we delete it
train = train.drop(['FireplaceQu'],axis=1)

# If na rate is lower then 20% we consider filling them with mean
# Since there are two types of variables with na rate lower than 1%,
# one is related to garage, another is related to basement
# Therefore, we create two new variables to indicate the presence of 
# garage and basement
train['hasGarage'] = train['GarageYrBlt'].notnull().astype(int)
train['hasBasement'] = train['BsmtQual'].notnull().astype(int)
train = train.drop(['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
                    'BsmtExposure','BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1'],axis=1)
train['MasVnrType'] = train['MasVnrType'].fillna('None')
train['MasVnrArea'] = train['MasVnrArea'].fillna(0)
train['Electrical'] = train['Electrical'].fillna(train['Electrical'].mode()[0])

get_na_rate(train)      # Checks if there exists NA

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

# Convert numerical data to categorical data 
categorical = ['MSSubClass','OverallQual','OverallCond','YrSold','MoSold']
for col in categorical:
    train[col] = train[col].astype(str)

qualitative = [qual for qual in train.columns if train.dtypes[qual] == 'O']
quantitative = [quan for quan in train.columns if quan not in qualitative]
print("qualitative: {}, quantitative: {}" .format (len(qualitative),len(quantitative)))
for col in qualitative:
    le = LabelEncoder()
    le.fit(list(train[col].values))
    train[col] = le.transform(list(train[col].values))

train.shape
# train.to_csv('encoder.csv',index=False)

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

y_train = np.log(train['SalePrice'])
x_train = train.drop(['SalePrice','Id'],axis = 1)

# df_dummies_train = pd.get_dummies(train)
# df_dummies_train == train
# train.to_csv('train_after_cleaning.csv',index=False)

y_train.to_csv('JX_train_y.csv',index=False)
x_train.to_csv('JX_train_x.csv',index=False)

########################### Building Model ##################################
# Source: https://www.kaggle.com/vjgupta/reach-top-10-with-simple-model-on-housing-prices/notebook

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
#import xgboost as xgb
import lightgbm as lgb

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

LassoMd = lasso.fit(x_train.values,y_train)
ENetMd = ENet.fit(x_train.values,y_train)
GBoostMd = GBoost.fit(x_train.values,y_train)

############################### Test ####################################

df_na = get_na_rate(raw_test)
drop_names = list(df_na[df_na.percent > 0.6].index) 
test = raw_test.drop(drop_names, axis=1)
get_na_rate(test)
test = test.drop(['FireplaceQu'],axis=1)

test['hasGarage'] = test['GarageYrBlt'].notnull().astype(int)
test['hasBasement'] = test['BsmtQual'].notnull().astype(int)
test = test.drop(['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
                    'BsmtExposure','BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1'],axis=1)
test['MasVnrType'] = test['MasVnrType'].fillna('None')
test['MasVnrArea'] = test['MasVnrArea'].fillna(0)
fill_mean = ['LotFrontage','GarageArea','BsmtUnfSF','TotalBsmtSF',]
fill_mode = ['MSZoning','BsmtFullBath','Utilities','Functional','BsmtHalfBath','GarageCars',
             'KitchenQual','BsmtFinSF2','BsmtFinSF1','Exterior2nd','Exterior1st','SaleType']

for col in fill_mean:
    test[col] = test[col].fillna(test[col].mean())
    
for col in fill_mode:
    test[col] = test[col].fillna(test[col].mode()[0])

get_na_rate(test)

categorical = ['MSSubClass','OverallQual','OverallCond','YrSold','MoSold']
for col in categorical:
    test[col] = test[col].astype(str)

qualitative = [qual for qual in test.columns if test.dtypes[qual] == 'O']
quantitative = [quan for quan in test.columns if quan not in qualitative]
print("qualitative: {}, quantitative: {}" .format (len(qualitative),len(quantitative)))

for col in qualitative:
    le = LabelEncoder()
    le.fit(list(test[col].values))
    test[col] = le.transform(list(test[col].values))

test.shape

for col in log_cols:
    test[col] = np.log1p(test[col].values)

df_zero = get_zero_rate(test).head(30)
df_zero

drop_cols = list(df_zero[df_zero.percent > 0.9].index) 
test = test.drop(drop_cols, axis=1)

# For those columns with percentage of zero about 50%, convert them to the dummy variable
def convert_to_boolean_test(column):
    return test[column].apply(lambda x: 0 if x > 0 else 1)

test['has2ndFlr'] = convert_to_boolean_test('2ndFlrSF')
test['hasWoodDeck'] = convert_to_boolean_test('WoodDeckSF')
test['hasOpenPorch'] = convert_to_boolean_test('OpenPorchSF')

test = test.drop(['Id'],axis=1)
test.shape
x_test = test[x_train.columns]

x_test.to_csv('JX_test.csv',index=False)

########################### Building Model ##################################
# Source: https://www.kaggle.com/vjgupta/reach-top-10-with-simple-model-on-housing-prices/notebook

finalMd = (np.expm1(LassoMd.predict(x_test.values)) + np.expm1(ENetMd.predict(x_test.values)) + np.expm1(GBoostMd.predict(x_test.values)) ) / 3
finalMd

result = np.expm1(LassoMd.predict(x_test.values))

predictions = pd.DataFrame()
predictions['Id'] = raw_test['Id']
predictions['SalePrice'] = finalMd
predictions.to_csv('lasso+Enet+Gboost.csv',index=False)