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

raw_train = pd.read_csv('train.csv')   
raw_test = pd.read_csv('test.csv')

x_train = raw_train.drop(['SalePrice'],axis=1)
y_train = raw_train['SalePrice']

all_data = pd.concat([x_train,raw_test],ignore_index=True)

# 1. Explore the SalePrice and correlation

# Top 10 heat map
k = 10
corrmat = raw_train.corr()
top = corrmat.nlargest(k, 'SalePrice').index  # Find the index of top 10 highest correlated variables
top_mat = corrmat.loc[top, top]               # Location their postions in heat map by index
fig, ax = plt.subplots(figsize=(8, 6))
sns.set(font_scale=1.25)
sns.heatmap(top_mat, annot=True, annot_kws={'size': 12}, square=True)
plt.show()

# 2. Explore the Saleprice

y_train.describe()

sns.distplot(y_train)
fig = plt.figure()
stats.probplot(y_train, plot=plt)

# Normalize the Saleprice with log transformation
y_train = np.log(y_train)
sns.distplot(y_train)
fig = plt.figure()
stats.probplot(y_train, plot=plt)

# 3. Check the na rate
def get_na_rate(dataframe):
    na_count = dataframe.isnull().sum()
    na_rate = na_count / len(dataframe)
    na_df = pd.concat([na_count, na_rate], axis=1, keys=['count', 'percent'])
    na_df = na_df[na_df['percent']>0]
    na_df = na_df.sort_values(['percent'], ascending=False)
    return na_df

df_na = get_na_rate(all_data)
df_na

# If na rate is higher than 0.6, then drop the column
drop_names = list(df_na[df_na.percent > 0.6].index) 
all_data = all_data.drop(drop_names, axis=1)

# FireplaceQu's na rate is about 50% but we already have another feature "Fireplaces"
# Therefre we delete it
all_data = all_data.drop(['FireplaceQu'],axis=1)

# If na rate is lower then 20% we consider filling them with mean
# Since there are two types of variables with na rate lower than 1%,
# one is related to garage, another is related to basement
# Therefore, we create two new variables to indicate the presence of 
# garage and basement
all_data['hasGarage'] = all_data['GarageYrBlt'].notnull().astype(int)
all_data['hasBasement'] = all_data['BsmtQual'].notnull().astype(int)
all_data = all_data.drop(['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
                    'BsmtExposure','BsmtFinType2','BsmtQual','BsmtCond','BsmtFinType1'],axis=1)
    
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

fill_mean = ['LotFrontage','GarageArea','BsmtUnfSF','TotalBsmtSF',]
fill_mode = ['MSZoning','BsmtFullBath','Utilities','Functional','BsmtHalfBath','GarageCars',
             'KitchenQual','BsmtFinSF2','BsmtFinSF1','Exterior2nd','Exterior1st','SaleType','Electrical']

for col in fill_mean:
    all_data[col] = all_data[col].fillna(all_data[col].mean())
    
for col in fill_mode:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
# Double check if there is NA   
df_na = get_na_rate(all_data)
df_na

# 4. Classify the data to the qualitative and quantitative data

# Convert numerical data to categorical data 
categorical = ['MSSubClass','OverallQual','OverallCond','YrSold','MoSold']
for col in categorical:
    all_data[col] = all_data[col].astype(str)

qualitative = []
quantitative = []
for col in all_data.columns:
    if all_data.dtypes[col] == 'O':
        qualitative.append(col)
    else:
        quantitative.append(col)   
print("qualitative: {}, quantitative: {}" .format (len(qualitative),len(quantitative)))

all_data.shape

# Have a look at the skewness of quantitative data
melt = pd.melt(all_data, value_vars=quantitative)
splot = sns.FacetGrid(melt, col="variable",  col_wrap=4, sharex=False, sharey=False)
splot.map(sns.distplot, "value")

df_skew = pd.DataFrame({'Variable':quantitative, 'Skewness':skew(all_data[quantitative])})
df_skew = df_skew.sort_values(by=['Skewness'], ascending=False)
df_skew

col_skew = df_skew[abs(df_skew.Skewness) > 0.75].Variable
all_data[col_skew] = np.log1p(all_data[col_skew])

def get_zero_rate(dataframe):
    zero_count = len(dataframe) - dataframe.astype(bool).sum(axis=0)
    zero_rate = zero_count / len(dataframe)
    zero_df = pd.concat([zero_count, zero_rate], axis=1, keys=['count', 'percent'])
    zero_df = zero_df.sort_values(['percent'],ascending=False)
    return zero_df

df_zero = get_zero_rate(all_data).head(20)
df_zero

# If one column has too many zero ( > 90%), then delete the columns
drop_cols = list(df_zero[df_zero.percent > 0.9].index) 
all_data = all_data.drop(drop_cols, axis=1)

# For those columns with percentage of zero about 50%, convert them to the dummy variable
def convert_to_boolean(column):
    return all_data[column].apply(lambda x: 0 if x > 0 else 1)

all_data['has2ndFlr'] = convert_to_boolean('2ndFlrSF')
all_data['hasWoodDeck'] = convert_to_boolean('WoodDeckSF')
all_data['Fireplaces'] = convert_to_boolean('Fireplaces')
all_data['hasOpenPorch'] = convert_to_boolean('OpenPorchSF')

all_data.shape

# Convert the categorical data to dummy variables
dummies_data = pd.get_dummies(all_data)

dummies_data.shape

x_train = dummies_data.iloc[0:len(x_train),]
x_test = dummies_data.iloc[len(x_train):,]

x_train = x_train.drop(['Id'],axis=1)
x_test = x_test.drop(['Id'],axis=1)

x_train.shape
x_test.shape

x_train.to_csv('x_train_dummies.csv',index=False)
x_test.to_csv('x_test_dummies.csv',index=False)