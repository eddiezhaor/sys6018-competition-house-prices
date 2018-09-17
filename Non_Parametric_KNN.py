
# coding: utf-8

# In[220]:


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


# In[241]:


#import training data set
rawData=pd.read_csv("./all/train.csv")
#import test data set
testData = pd.read_csv("./all/test.csv")
cleanTrain = pd.read_csv("../Jiangxue_code/JX_train_x.csv")
cleanTest = pd.read_csv("../Jiangxue_code/JX_test.csv")
dummyTraining = pd.read_csv("../Jiangxue_code/x_train_dummies.csv")
corrmat = rawData.corr()
top = corrmat.nlargest(50, 'SalePrice').index  # Find the index of top 10 highest correlated variables
d1 = pd.read_csv("x_train_no_log_no_sd.csv")
d2 =pd.read_csv("x_test_no_log_no_sd.csv")
salePrice = rawData.SalePrice 
newData = rawData[top[0:15]] #select the top 10 highest correlated variables
newData.GarageYrBlt = newData.GarageYrBlt.fillna(value = newData.GarageYrBlt.median()) #fill missing data
newData.MasVnrArea = newData.MasVnrArea.fillna(0)
newData2 = newData.copy()
# select attributes
dataNew = SelectKBest(chi2,11).fit(newData2, salePrice) #use 
newData3 = newData2.iloc[:,dataNew.get_support()]
# newData3 = newData3[newData3.SalePrice<650000]
# newData3.iloc[:,1:10] = np.log(1+newData3.iloc[:,1:10])
newData3.iloc[:,1:11] = preprocessing.StandardScaler().fit_transform(newData3.iloc[:,1:11]) #standardize the data
colName = newData3.iloc[:,1:11].columns
salePrice = np.log(1+np.array(newData3.SalePrice)) #logrithms
newData3= np.array(newData3.iloc[:,1:11]) #convert the data into matrix 


# In[242]:


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
        


# In[243]:


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
#print out the average RSMLE
print(np.array(AvgRsmle).mean())


# In[244]:


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
    #create a weight matrix using the inverse of the distance
    weights = weightsMatrix[:,:k]
    weights[weights==0] =1
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
        


# In[245]:


kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
X = newData3
y = salePrice
#K-fold cross validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(manhantanKNN(X_test, X_train,y_train, 10,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[246]:


#KNN with manhantan distance and gaussian kernel
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


# In[247]:


#K-fold cross validation
kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
X = newData3
y = salePrice
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(KernelKNN(X_test, X_train,y_train, 8,0.1,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[248]:


# Predmean = KernelKNN(test3, X,y,15,0.1,"Y")
# myId = pd.DataFrame(testData.Id)
# prediction = pd.DataFrame(Predmean.reshape(-1,1), columns=["SalePrice"])

# prediciton2 = pd.concat([myId,prediction], axis=1,sort=False)
# prediciton2.to_csv("Mysubmission1.csv",index=False) #output it into a csv file


# In[249]:


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
    
    


# In[250]:


#K-fold cross validation
kf = KFold(n_splits=10,random_state=None, shuffle=True)
AvgRsmle = []
X = newData3
y = salePrice
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    AvgRsmle.append(euclideanKNN(X_test, X_train,y_train, 10,0.1,"N",y_test))
print(np.array(AvgRsmle).mean())


# In[252]:


#we choose weighted KNN with manhantan distance which has the highest accuracy compared to other 3 knn performances for prediction
test1 = testData[colName]
test2 = test1.fillna(test1.mean())
test3 = preprocessing.StandardScaler().fit_transform(test2)
newPred = []
#bagging
for i in range(8,16):
    newPred.append(manhantanKNN(test3, X,y,i,"Y"))
Predmean = np.array(newPred).mean(axis=0)
#convert back to normal numbers from log10
Predmean = np.exp(Predmean)-1
myId = pd.DataFrame(testData.Id)
prediction = pd.DataFrame(Predmean.reshape(-1,1), columns=["SalePrice"])
prediciton2 = pd.concat([myId,prediction], axis=1,sort=False)
prediciton2.to_csv("finalSubmission.csv",index=False) #output it into a csv file

