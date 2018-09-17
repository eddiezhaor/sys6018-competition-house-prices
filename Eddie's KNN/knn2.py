
# coding: utf-8

# In[1386]:


import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[1884]:


#import training data set
rawData=pd.read_csv("./Eddie's KNN/all/train.csv")
#import test data set
testData = pd.read_csv("./Eddie's KNN/all/test.csv")
cleanTrain = pd.read_csv("./Jiangxue_code/JX_train_x.csv")
cleanTest = pd.read_csv("./Jiangxue_code/JX_test.csv")
dummyTraining = pd.read_csv("./Jiangxue_code/x_train_dummies.csv")
corrmat = rawData.corr()
top = corrmat.nlargest(50, 'SalePrice').index  # Find the index of top 10 highest correlated variables
salePrice = rawData.SalePrice 
newData = rawData[top[0:14]] #select the top 10 highest correlated variables

newData.GarageYrBlt = newData.GarageYrBlt.fillna(value = newData.GarageYrBlt.median())
newData.MasVnrArea = newData.MasVnrArea.fillna(0)
min_max_scale = preprocessing.MinMaxScaler()
newData2 = newData.copy()
# # and select attributes
dataNew = SelectKBest(chi2,9).fit(newData2, salePrice) #use 
# newData2["newQuality"] = newData2.OverallQual**2
# newData2["OverallQual"] = newData2["OverallQual"]*500
# feature = dataNew.transform(newData2)
newData3 = newData2.iloc[:,dataNew.get_support()]

# newData3["OverallQual"] = newData3["OverallQual"]*500
# newData3 = min_max_scale.fit_transform(newData3) #normalize data
# newData3 = newData3[newData3.SalePrice<700000]
newData3 = np.log(1+newData3)
newData3.iloc[:,1:9] = preprocessing.StandardScaler().fit_transform(newData3.iloc[:,1:9])
print(newData3)
# for i in range(newData3.shape[1]):
#     newData3 = newData3[newData3[:,i]<=6,:]
# # newData3 = min_max_scale.fit_transform(newData3) #normalize data
newData3 = np.array(newData3)
# for i in range(1,newData3.shape[1]):
#     newData3 = newData3[newData3[:,i]<=5.8,:]
salePrice = newData3[:,0]
newData3= newData3[:,1:9]
# salePrice = np.log(1+salePrice)


# In[1880]:


#create a func to calulate the rmsle 
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
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
        return(housePrice.mean(axis=1))
    elif prediction == "N":
        return rmsle(housePrice.mean(axis=1),testSet)
        


# In[1886]:


from sklearn.model_selection import KFold
X = newData3
y = salePrice
kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
#K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(uniformKNN(X_test, X_train,y_train,8,"N",y_test))
print(np.array(AvgRsmle).mean())
test1 = testData[top[1:14]]
test2 = test1.iloc[:,dataNew.get_support()[1:14]]
test2 = test2.fillna(test2.mean())
# test2["OverallQual"] = test2["OverallQual"]*1000
test3 = preprocessing.StandardScaler().fit_transform(test2)
# Predmean = uniformKNN(test3, X,y,8,"Y")
# myId = pd.DataFrame(testData.Id)
# prediction = pd.DataFrame(Predmean.reshape(-1,1), columns=["SalePrice"])

# prediciton2 = pd.concat([myId,prediction], axis=1,sort=False)
# prediciton2.to_csv("Mysubmission2.csv",index=False) #output it into a csv file


# In[1882]:


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
    weights =weights+0.5
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
        


# In[1883]:


test1 = testData[top[1:14]]
test2 = test1.iloc[:,dataNew.get_support()[1:14]]
test2 = test2.fillna(test2.mean())
# test2["OverallQual"] = test2["OverallQual"]*1000
test2 = np.log(1+test2)
test3 = preprocessing.StandardScaler().fit_transform(test2)
X = newData3
y = salePrice
kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []

#K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(manhantanKNN(X_test, X_train,y_train, 8,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[1877]:


newPred = []
#bagging
for i in range(8,16):
    newPred.append(manhantanKNN(test3, X,y,i,"Y"))
Predmean = np.array(newPred).mean(axis=0)
Predmean = np.exp(Predmean)-1


# In[1878]:


# Predmean = manhantanKNN(test3, X,y,15,"Y")
myId = pd.DataFrame(testData.Id)
prediction = pd.DataFrame(Predmean.reshape(-1,1), columns=["SalePrice"])
prediciton2 = pd.concat([myId,prediction], axis=1,sort=False)
prediciton2.to_csv("Mysubmission6.csv",index=False) #output it into a csv file


# In[1821]:


#Weighted KNN with manhantan distance and gaussian kernel
def gaussianKernel(distance,lamda):
    return np.exp(-(distance**2)/lamda)
def KernelKNN(myTest, trainingData, Y_train, k, lamda,prediction, testSet=None):
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
    weights =weights
    newWeights = gaussianKernel(weights,lamda)
    Y_train = Y_train.reshape(1,-1)
    #query the house price
    housePrice = Y_train[0][getIndex2]
    weightedHouseprice = newWeights*housePrice
    if prediction == "Y":
        #take the average saleprice
        return (weightedHouseprice.sum(axis=1)/np.sum(newWeights,axis=1))
    elif prediction == "N":
        return rmsle(weightedHouseprice.sum(axis=1)/np.sum(newWeights,axis=1),testSet)


# In[1823]:



kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
#K-fold cross validation
X = newData3
y = salePrice
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(KernelKNN(X_test, X_train,y_train, 8,0.1,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[1660]:


# Predmean = KernelKNN(test3, X,y,15,0.1,"Y")
# myId = pd.DataFrame(testData.Id)
# prediction = pd.DataFrame(Predmean.reshape(-1,1), columns=["SalePrice"])

# prediciton2 = pd.concat([myId,prediction], axis=1,sort=False)
# prediciton2.to_csv("Mysubmission1.csv",index=False) #output it into a csv file


# In[1824]:


#KNN with euclidean distance and gaussian kernel
def euclideanKNN(myTest, trainingData, Y_train, k,lamda, prediction, testSet=None):
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
    weightsMatrix = np.sort(array2)
    weights = weightsMatrix[:,:k]
    weights =weights
#     weights = weights**(-1)
    newWeights = gaussianKernel(weights,lamda)
    Y_train = Y_train.reshape(1,-1)
    #query the house price
    housePrice = Y_train[0][getIndex2]
    weightedHouseprice = newWeights*housePrice
    if prediction == "Y":
        #take the average saleprice
        return (weightedHouseprice.sum(axis=1)/np.sum(newWeights,axis=1))
    elif prediction == "N":
        return rmsle(weightedHouseprice.sum(axis=1)/np.sum(newWeights,axis=1),testSet)
    
    


# In[1825]:


kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
X = newData3
y = salePrice
#K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(euclideanKNN(X_test, X_train,y_train, 10,0.1,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[ ]:




