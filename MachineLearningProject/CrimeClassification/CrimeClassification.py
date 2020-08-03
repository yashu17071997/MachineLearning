# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:22:37 2020

@author: HP
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
def saveData():
    traindata=pd.read_csv(trainSetIP)
    testdata=pd.read_csv(testSetIP)
    traindata.drop(["Resolution","Descript","Category","Address"],axis=1,inplace=True)
    testdata.drop(["Id","Address"],axis=1,inplace=True)
    #print(type(traindata))
    #print(type(testdata))
    somelistTr=[]
    for cnt in range(0,len(traindata)):
        somelistTr.append(traindata["Dates"][cnt].split()[1][0:5])
    traindata["years"]=somelistTr
    
    somelistTs=[]
    for cnt in range(0,len(testdata)):
        somelistTs.append(testdata["Dates"][cnt].split()[1][0:5])
    testdata["years"]=somelistTs
    
    coordinatesTr=[]
    for cnt in range(0,len(traindata)):
        coordinatesTr.append(str(traindata["X"][cnt])+str(traindata["Y"][cnt]))
    traindata["coordinates"]=coordinatesTr
    
    coordinatesTs=[]
    for cnt in range(0,len(testdata)):
        coordinatesTs.append(str(testdata["X"][cnt])+str(testdata["Y"][cnt]))
    testdata["coordinates"]=coordinatesTs
    #print(testdata.head(10))
    #print(traindata.head(10))
    testdata.drop(["Dates"],axis=1,inplace=True)
    traindata.drop(["Dates"],axis=1,inplace=True)
    testdata.drop(["X","Y"],axis=1,inplace=True)
    traindata.drop(["X","Y"],axis=1,inplace=True)
    uniqueCTr=list(set(traindata["coordinates"]))
    uniqueCTr.sort
    uniqueCTs=list(set(testdata["coordinates"]))
    uniqueCTs.sort
    uniqueDoWTr=list(set(traindata["DayOfWeek"]))
    uniqueDoWTr.sort
    uniquePdDTr=list(set(traindata["PdDistrict"]))
    uniquePdDTr.sort
    uniqueyearsTr=list(set(traindata["years"]))
    uniqueyearsTr.sort
    #['CENTRAL', 'RICHMOND', 'INGLESIDE', 'TENDERLOIN', 
    #'BAYVIEW', 'TARAVAL', 'MISSION', 'SOUTHERN', 'NORTHERN', 'PARK']
    #print(testdata.head(10))
    #print(traindata.head(10))
    uniquePdDTr=list(set(traindata["PdDistrict"]))
    uniquePdDTr.sort
    #print(uniquePdDTr)
    for each in range(0,len(uniqueCTr)):
        traindata["coordinates"].replace({uniqueCTr[each]:each+1},inplace=True)
    for each in range(0,len(uniqueCTs)):
        testdata["coordinates"].replace({uniqueCTs[each]:each+1},inplace=True)
    for each in range(0,len(uniqueyearsTr)):
        traindata["years"].replace({uniqueyearsTr[each]:each+1},inplace=True)
        testdata["years"].replace({uniqueyearsTr[each]:each+1},inplace=True)
    for each in range(0,len(uniqueDoWTr)):
        traindata["DayOfWeek"].replace({uniqueDoWTr[each]:each+1},inplace=True)
        testdata["DayOfWeek"].replace({uniqueDoWTr[each]:each+1},inplace=True)    
    for i in classToIncludeList:
        testdata=testdata[testdata.PdDistrict!=i]
        traindata=traindata[traindata.PdDistrict!=i]    
    #print(uniquePdDTr)
    #print(testdata.head(10))
    #print(traindata.head(10))    
    uniquePdDTr=list(set(traindata["PdDistrict"]))
    uniquePdDTr.sort
    #print(uniquePdDTr)
    for each in range(0,len(uniquePdDTr)):
        traindata["PdDistrict"].replace({uniquePdDTr[each]:each+1},inplace=True)
        testdata["PdDistrict"].replace({uniquePdDTr[each]:each+1},inplace=True)
    #outputTrFile=input(r"Enter a file name for train data output Ex: D:\\dataframeTrainSet.csv")
    #traindata.to_csv(r'D:\\san-francisco-crime-classification\\dataframeTrainSet.csv', index = False, header=True)
    traindata.to_csv(outputTrFile, index = False, header=True)
    #outputTsFile=input(r"Enter a file name for test data output Ex: D:\\dataframeTestSet.csv")
    #testdata.to_csv(r'D:\\san-francisco-crime-classification\\dataframeTestSet.csv', index = False, header=True)
    testdata.to_csv(outputTsFile, index = False, header=True)


global outputTrFile
global outputTsFile
global trainSetIP
global testSetIP
print("Classes available: CENTRAL,RICHMOND,INGLESIDE,TENDERLOIN,BAYVIEW,TARAVAL,MISSION,SOUTHERN,NORTHERN,PARK")
print()
print("Enter the classes you DO NOT need: ")
classToInclude=input("Must be uppercase, no spaces and must be seperated with comma as ahown above. For example to NOT include Central and Richmond Type: CENTRAL,RICHMOND")
classToIncludeList=classToInclude.split(",")
#D:\\san-francisco-crime-classification\\train.csv
#D:\\san-francisco-crime-classification\\test.csv
trainSetIP=input("Enter the path of trainset of input data file ")
#trainSetIP="D:\\san-francisco-crime-classification\\train.csv"
testSetIP=input("Enter the path of testset of input data file ")
#testSetIP="D:\\san-francisco-crime-classification\\test.csv"
outputTrFile=input(r"Enter an output csv file name for train data output Ex: D:\\dataframeTrainSet.csv ")
#outputTrFile=r"D:\\san-francisco-crime-classification\\dataframeTrainSetcl10.csv"
outputTsFile=input(r"Enter an output csv file name for test data output Ex: D:\\dataframeTestSet.csv ")
#outputTsFile=r"D:\\san-francisco-crime-classification\\dataframeTestSetcl10.csv"
print("Running...")
saveData()
#D:\\san-francisco-crime-classification\\dataframeTrainSet1.csv
trainData=pd.read_csv(outputTrFile)
#D:\\san-francisco-crime-classification\\dataframeTestSet1.csv
testData=pd.read_csv(outputTsFile)
trainX=trainData[trainData.columns[trainData.columns!='PdDistrict']].values
trainY=trainData['PdDistrict'].values
testX=testData[testData.columns[testData.columns!='PdDistrict']].values
testY=testData['PdDistrict'].values

#Naive bayes
nb=GaussianNB()
nb.fit(np.array(trainX), trainY)
yPred=nb.predict(testX)
print("Naive Bayesian score",nb.score(testX, testY))
mse=metrics.mean_squared_error(testY,yPred)
#print("MSE using Naive Bayes:", mse)
print("Accuracy using Naive Bayes:", metrics.accuracy_score(testY,yPred))

print()
print()
#KNN
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(trainX,trainY)
yPred=knn.predict(testX)
mse=metrics.mean_squared_error(testY,yPred)
print("KNN score:",knn.score(testX, testY))
#print("MSE using KNN:", mse)
testScores={}
allScores=[]
for k in range(1,101):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(trainX,trainY)
    yPred=knn.predict(testX)
    testScores[k]=metrics.accuracy_score(testY,yPred)
    allScores.append(metrics.accuracy_score(testY,yPred))
print("Max accuracy of KNN:",max(allScores),"at k=",allScores.index(max(allScores)))

print()
print()
#Linear regression
linearRegression=linear_model.LinearRegression()
linearRegression.fit(trainX,trainY)
yPred=linearRegression.predict(testX)
mse=metrics.mean_squared_error(testY,yPred)
mae=metrics.mean_absolute_error(testY,yPred)
rmse=np.sqrt(metrics.mean_squared_error(testY,yPred))
print("Linear regression score",linearRegression.score(testX,testY))
#print("MSE using linear regression:", mse)
print("Mean squared error:",mse)
print("Mean absolute error:",mae)
print("Root mean squared error:",rmse)

print()
print()
#Decision tree
dtclf=DecisionTreeClassifier()
dtclf.fit(trainX,trainY)
yPred=dtclf.predict(testX)
mse=metrics.mean_squared_error(testY,yPred)
print("Decision tree score",dtclf.score(testX,testY))
#print("MSE using decision tree:", mse)
print("Max accuracy of Decision tree classifier:",metrics.accuracy_score(testY, yPred))

print()
print()
#Gradient boosting
args={'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
gbclf=ensemble.GradientBoostingRegressor(**args)
gbclf.fit(trainX,trainY)
yPred=gbclf.predict(testX)
print("Gradient boosting score",gbclf.score(testX,testY))
mse=metrics.mean_squared_error(testY,yPred)
#print("MSE using gradient boosting:", mse)
accScore=np.zeros((args['n_estimators'],), dtype=np.float64)
for i, predY in enumerate(gbclf.staged_predict(testX)):
    accScore[i]=gbclf.loss_(testY,predY)
print("Max accuracy of Gradient Boosting:",max(accScore),"at ",int(np.where(accScore==(max(accScore)))[0])+1)

print()
print()
#Logistic regresssion
logisticRegression=LogisticRegression(max_iter=1000,random_state=0,solver='lbfgs',multi_class='auto')
logisticRegression.fit(trainX,trainY)
yPred=logisticRegression.predict(testX)
mse=metrics.mean_squared_error(testY,yPred)
print("Logistic regression score",logisticRegression.score(testX,testY))
#print("MSE using Logistic regression:", mse)
print("Accuracy of Logistic regression",metrics.accuracy_score(testY,yPred))

print()
print()
#Random forest
rfclf=RandomForestClassifier(n_estimators=100)
rfclf.fit(trainX,trainY)
yPred=rfclf.predict(testX)
mse=metrics.mean_squared_error(testY,yPred)
print("Random forest score",rfclf.score(testX,testY))
#print("MSE using random forest:", mse)
print("Accuracy using random forest:",metrics.accuracy_score(testY,yPred))