import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,RidgeCV,LassoCV

# For parametric approach, we tried to use linear regression with 10 highest-correlated
# variables at first, but the performance based on cross validation is not good
# Therefore we tried ridge regression and lasso regression for prediction,
# the the performance is improved.
# Therefore our final prediction is based on the combination of ridge regression
# and lasso regression.

# Import all the datasets
raw_test = pd.read_csv('test.csv')
x_train_org = pd.read_csv('x_train_dummies.csv')
y_train_org = pd.read_csv('y_train.csv',header=None,names=['SalePrice'])
x_test = pd.read_csv('x_test_dummies.csv')
    
# Split datasets into training set and validation set
x_train, x_cv, y_train, y_cv = train_test_split(x_train_org, y_train_org,test_size=0.33, random_state=42)

# 1. Linear Regression
all_train = pd.concat([x_train,y_train],axis=1)  # Merges variables and target variable into one dataset
corrmat = all_train.corr()                    
top = corrmat.nlargest(10, 'SalePrice').index    # Gets the top 10 highest correlated variables

lreg = LinearRegression()
lreg.fit(x_train[list(top[1:])],y_train)         # Uses the top 10 highest correlated variables for prediction
pred = lreg.predict(x_cv[list(top[1:])])         # Uses the validation set for prediction
mse = np.mean((pred - y_cv.values)**2)           # Calculates the mean squared error
mse
# 0.03245792559968048


# 2. Ridge Regression
# Select the best alpha using the entire training set
alphas = np.arange(0.01,10,0.01)    
ridge_cv = RidgeCV(alphas=alphas)
cv_score = ridge_cv.fit(x_train_org, y_train_org)
cv_score.alpha_
# 5.08

# Train ridge model with the training dataset after cross validation
ridge = Ridge(alpha=5.33)
ridge.fit(x_train,y_train)
pred = ridge.predict(x_cv)                      # Uses the validation set for prediction
mse = np.mean((pred - y_cv.values)**2)          # Calculates the mean squared error
mse                                             
# 0.01647050684221818

# 3. Lasso Regression
# Select the best alpha using the entire training set
lasso_cv = LassoCV(alphas=alphas)
cv_score = lasso_cv.fit(x_train_org, y_train_org)
cv_score.alpha_
# 0.01

# Train lasso model with the training dataset after cross validation
lasso = Lasso(alpha=0.01, random_state=1)
lasso.fit(x_train,y_train)
pred = lasso.predict(x_cv)                      # Uses the validation set for prediction
mse = np.mean((pred - y_cv.values)**2)          # Calculates the mean squared error
mse
# 0.2890502578714499

# Train the models based on entire training dataset
ridge.fit(x_train_org,y_train_org)
lasso.fit(x_train_org,y_train_org)

# Predict on test dataset
# Since the target variable used in training has been log transformed,
# therefore we should calculate the result with exponential function.
df_ridge = pd.DataFrame(np.expm1(ridge.predict(x_test)))
df_lasso = pd.DataFrame(np.expm1(lasso.predict(x_test)))

# Since the performance of ridge regression is better than lasso,
# therefore we weight the result from ridge regression more.
result = 0.7*df_ridge + 0.3*df_lasso  

# Export the final result to csv file.
predictions = pd.DataFrame()
predictions['Id'] = raw_test['Id']
predictions['SalePrice'] = result
predictions.to_csv('parametric-approach-result.csv',index=False)

