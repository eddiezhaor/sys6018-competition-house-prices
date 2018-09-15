
# coding: utf-8

# In[396]:


import pandas as pd
import numpy as np
import math 


def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))


#import training data set
rawData=pd.read_csv("./all/train.csv")
salePrice1 = rawData[["SalePrice"]]
#select features
myData = rawData[["MoSold","LotArea"]]
#import test data set
testData = pd.read_csv("./all/test.csv")
myTest1 = testData[["MoSold","LotArea"]]
def uniformKNN(myTest, trainingData, Y_train, k, prediction, testSet=None):
    testLength = myTest.shape[0]
    trainLenth = trainingData.shape[0]
    #convert it into a matrix 
    Y_train = np.array(Y_train)
    trainingData = np.array(trainingData)
    myTest = np.array(myTest)
    #matrix substraction and reshape
    newArray = ((myTest[:,np.newaxis] - trainingData)**2).reshape(-1, myTest.shape[1])
    #sum up each row
    newArray2 = newArray.sum(axis=1)
    #take square root
    newArray2 = newArray2**(1/2)
    #reshape the matrix
    array2 = newArray2.reshape(testLength,trainLenth)
    #sort each row and display by index
    getIndex = np.argsort(array2)
    #take the K nearest neighbors
    getIndex2 = getIndex[:,:k]
    Y_train = Y_train.reshape(1,-1)
    #query the house price
    housePrice = Y_train[0][getIndex2]
    if prediction == "Y":
        #take the average saleprice
        print(housePrice.mean(axis=1))
    elif prediction == "N":
        return rmsle(housePrice.mean(axis=1),testSet)
        
    
uniformKNN(myTest1, myData, salePrice1,40,"Y")


# In[399]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
corrmat = rawData.corr()
top = corrmat.nlargest(20, 'SalePrice').index  # Find the index of top 10 highest correlated variables
salePrice = rawData.SalePrice 
newData = rawData[top[1:14]] #select the top 10 highest correlated variables

newData.GarageYrBlt = newData.GarageYrBlt.fillna(value = newData.GarageYrBlt.median())
newData.MasVnrArea = newData.MasVnrArea.fillna(0)
print(newData)
min_max_scale = preprocessing.MinMaxScaler()
newData2 = newData.copy()
dataNew = SelectKBest(chi2,8).fit(newData2, salePrice) #use 
newData3 = newData2.iloc[:,dataNew.get_support()]
newData3 = min_max_scale.fit_transform(newData3) #standarize the data


# In[400]:


from sklearn.model_selection import KFold
X = newData3
y = np.array(rawData["SalePrice"])
kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
#K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(uniformKNN(X_test, X_train,y_train,8,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[401]:


#Weighted KNN with manhantan distance
def manhantanKNN(myTest, trainingData, Y_train, k, prediction, testSet=None):
    testLength = myTest.shape[0]
    trainLenth = trainingData.shape[0]
    #convert it into a matrix 
    Y_train = np.array(Y_train)
    trainingData = np.array(trainingData)
    #matrix substraction and reshape

    newArray = (np.abs(myTest[:,np.newaxis] - trainingData)).reshape(-1,myTest.shape[1])
    #sum up each row
    newArray2 = newArray.sum(axis=1)
    #reshape the matrix
    array2 = newArray2.reshape(testLength,trainLenth)
    #sort each row and display by index
    getIndex = np.argsort(array2)
    #take the K nearest neighbors
    getIndex2 = getIndex[:,:k]
    weightsMatrix = np.sort(array2)
    weights = weightsMatrix[:,:k]
    weights =weights+1
    weights = weights**(-1)
    
    Y_train = Y_train.reshape(1,-1)
    #query the house price
    housePrice = Y_train[0][getIndex2]
    weightedHouseprice = weights*housePrice
    
    if prediction == "Y":
        #take the average saleprice
        return (weightedHouseprice.sum(axis=1)/np.sum(weights,axis=1))
    elif prediction == "N":
        return rmsle(weightedHouseprice.sum(axis=1)/np.sum(weights,axis=1),testSet)
        


# In[402]:


test1 = testData[top[1:14]]
test2 = test1.iloc[:,dataNew.get_support()]
test2 = test2.fillna(test2.mean())
# test2["newQuality"] = test2.OverallQual**2
# test2["newGara"] = test2.GarageArea*test2["1stFlrSF"]
test3 = min_max_scale.fit_transform(test2)
X = newData3
y = np.array(rawData["SalePrice"])

kf = KFold(n_splits=12,random_state=None, shuffle=True)
AvgRsmle = []
#K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(manhantanKNN(X_test, X_train,y_train, 8,"N",y_test))
print(np.array(AvgRsmle).mean())




Predmean = manhantanKNN(test3, X,y,8,"Y")
myId = pd.DataFrame(testData.Id)
prediction = pd.DataFrame(Predmean.reshape(-1,1), columns=["SalePrice"])

prediciton2 = pd.concat([myId,prediction], axis=1,sort=False)
prediciton2.to_csv("Mysubmission1.csv",index=False) #output it into a csv file


# In[ ]:




