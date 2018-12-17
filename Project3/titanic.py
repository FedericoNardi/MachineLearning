import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier


# Load TrainData
TrainData = pd.read_csv('Dataset/train.csv')
TestData = pd.read_csv('Dataset/test.csv')

"""
g = sns.FacetGrid(TrainData,col='Parch')
g.map(sns.distplot, 'Age', bins=20, norm_hist=True)
plt.show()
"""

#--------------------------------------------------------------------
# Fill holes
TrainData.info()

# Fill holes in Age
NaNIndex = list(TrainData['Age'][TrainData['Age'].isnull()].index)

for i in range(len(NaNIndex)):
	AgeMedian = TrainData['Age'].median()
	AgePredict = TrainData["Age"][ ((TrainData['Parch'] == TrainData.iloc[i]["Parch"]) & (TrainData['Pclass'] == TrainData.iloc[i]["Pclass"]) ) ] .median()
	if not np.isnan(AgePredict)==True:
		TrainData['Age'][NaNIndex[i]] = AgePredict
	else:
		TrainData['Age'][NaNIndex[i]] = AgeMedian

# Fill holes in Embarked
index = TrainData['Embarked'][TrainData['Embarked'].isnull()].index
for i in range(len(index)):	
	TrainData['Embarked'][index[i]] = 'S'


#----------------------------------------------------------------------
# Drop useless

# Create Family size variable
Family = TrainData['SibSp']+TrainData['Parch']
TrainData['Family'] = Family

#Drop not relevant data
TrainData.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'],axis=1,inplace=True)

TrainData.info()


#---------------------------------------------------------------------
# Convert categorical to int

TrainData['Embarked'] = TrainData['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(np.int64)
TrainData['Sex'] = TrainData['Sex'].map( {'male':0,'female':1} )
#print(TrainData['Embarked'].head())

#---------------------------------------------------------------------
# SETTING UP REGRESSIONS
# Split into train and test
Xdata = TrainData.drop("Survived",axis=1)
Ydata = TrainData['Survived']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xdata, Ydata, train_size=0.75)

# Logistic regression
Logistic = LogisticRegression()
Logistic.fit(Xtrain,Ytrain)
TrainPredict = Logistic.predict(Xtrain)
TestPredict = Logistic.predict(Xtest)

TrainAccuracy = Logistic.score(Xtrain,Ytrain)
TestAccuracy = Logistic.score(Xtest,Ytest)

print('='*40)
print('Logistic Regression')
print('Accuracy on training data: ',TrainAccuracy)
print('Accuracy on test data: ',TestAccuracy)


# Multilayer perceptron
MLP = Perceptron()
MLP.fit(Xtrain,Ytrain)
TrainPredict = MLP.predict(Xtrain)
TestPredict = MLP.predict(Xtest)

TrainAccuracy = MLP.score(Xtrain,Ytrain)
TestAccuracy = MLP.score(Xtest,Ytest)

print('='*40)
print('Multilayer Perceptron')
print('Accuracy on training data: ',TrainAccuracy)
print('Accuracy on test data: ',TestAccuracy)


# Random forest
RunForrest = RandomForestClassifier()
RunForrest.fit(Xtrain,Ytrain)
TrainPredict = RunForrest.predict(Xtrain)
TestPredict = RunForrest.predict(Xtest)

TrainAccuracy = RunForrest.score(Xtrain,Ytrain)
TestAccuracy = RunForrest.score(Xtest,Ytest)

print('='*40)
print('Random Forest')
print('Accuracy on training data: ',TrainAccuracy)
print('Accuracy on test data: ',TestAccuracy)
