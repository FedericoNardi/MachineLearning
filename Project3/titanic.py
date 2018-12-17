#%%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
<<<<<<< HEAD
from sklearn.neural_network import MLPClassifier
=======
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
>>>>>>> e60277935b352febf5589ed54f1397a72d106bd0
from sklearn.ensemble import RandomForestClassifier


# Load TrainData
TrainData = pd.read_csv('Dataset/train.csv')
TestData = pd.read_csv('Dataset/test.csv')

"""
g = sns.FacetGrid(TrainData,col='Parch')
g.map(sns.distplot, 'Age', bins=20, norm_hist=True)
plt.show()
"""
<<<<<<< HEAD
#--------------------------------------------------------------------

=======

#--------------------------------------------------------------------
>>>>>>> e60277935b352febf5589ed54f1397a72d106bd0
# Fill holes
TrainData.info()

# Fill holes in Embarked
index = TrainData['Embarked'][TrainData['Embarked'].isnull()].index
for i in range(len(index)):	
	TrainData['Embarked'][index[i]] = 'S'
#%%
# Convert categorical to int

TrainData['Embarked'] = TrainData['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(np.int64)
TrainData['Sex'] = TrainData['Sex'].map( {'male':0,'female':1} )


## Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
#sns.heatmap(TrainData[["Age","SibSp","Parch","Pclass", "Sex","Fare","Embarked"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

# Explore Age vs Survived
g = sns.FacetGrid(TrainData, col='Pclass')
g.map(plt.hist, 'Age', bins=20)

#sns.factorplot(y="Age",x="Pclass", data=TrainData,kind="box")
#sns.factorplot(y="Age",x="SibSp", data=TrainData,kind="box")

# Most frequent port
#freq_port = TrainData.Embarked.dropna().mode()[0]
#print("The most frequent port is")
#print(freq_port)


# Fill holes in Age
NaNIndex = list(TrainData['Age'][TrainData['Age'].isnull()].index)

for i in range(len(NaNIndex)):
	AgeMedian = TrainData['Age'].mean()
	AgePredict = TrainData["Age"][ ((TrainData['Parch'] == TrainData.iloc[i]["Parch"]) & (TrainData['Pclass'] == TrainData.iloc[i]["Pclass"]) ) ] .median()
	if not np.isnan(AgePredict)==True:
		TrainData['Age'][NaNIndex[i]] = AgePredict
	else:
		TrainData['Age'][NaNIndex[i]] = AgeMedian
<<<<<<< HEAD

# Fill holes in Embarked
index = TrainData['Embarked'][TrainData['Embarked'].isnull()].index
for i in range(len(index)):	
	TrainData['Embarked'][index[i]] = 'S'


#----------------------------------------------------------------------
=======


#----------------------------------------------------------------------
#%%
>>>>>>> e60277935b352febf5589ed54f1397a72d106bd0
# Drop useless

# Create Family size variable
Family = TrainData['SibSp']+TrainData['Parch']
TrainData['Family'] = Family

#Drop not relevant data
TrainData.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

TrainData.info()

<<<<<<< HEAD

#---------------------------------------------------------------------
# Convert categorical to int

TrainData['Embarked'] = TrainData['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(np.int64)
TrainData['Sex'] = TrainData['Sex'].map( {'male':0,'female':1} )


"""
#---------------------------------------------------------------------
=======
#---------------------------------------------------------------------
# Convert categorical to int
#print(TrainData['Embarked'].head())


>>>>>>> e60277935b352febf5589ed54f1397a72d106bd0
## Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
#sns.heatmap(TrainData[["Survived","Age","Family","Pclass", "Sex","Fare","Embarked"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")

g = sns.factorplot(x="Sex",y="Survived",data=TrainData,kind="bar", size = 6 , 
	palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Explore Pclass vs Survived
g = sns.factorplot(x="Pclass",y="Survived",data=TrainData,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Explore Embarked vs Survived 
g = sns.factorplot(x="Embarked", y="Survived",  data=TrainData,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

# Explore Embarked vs Survived 
g = sns.factorplot(x="Family", y="Survived",  data=TrainData,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


#print(TrainData['Embarked'].head())
"""

<<<<<<< HEAD
"""
=======
>>>>>>> e60277935b352febf5589ed54f1397a72d106bd0
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
MLP = MLPClassifier(activation='relu',solver='sgd',learning_rate='adaptive',verbose=True)
MLP.fit(Xtrain,Ytrain)
TrainPredict = MLP.predict(Xtrain)
TestPredict = MLP.predict(Xtest)

TrainAccuracy = MLP.score(Xtrain,Ytrain)
TestAccuracy = MLP.score(Xtest,Ytest)

print('='*40)
print('Multilayer Perceptron')
print('Accuracy on training data: ',TrainAccuracy)
print('Accuracy on test data: ',TestAccuracy)


# Support Vector Machines
svc = SVC()
svc.fit(Xtrain, Ytrain)
TrainPredict = svc.predict(Xtrain)
TestPredict = svc.predict(Xtest)

TrainAccuracy = svc.score(Xtrain,Ytrain)
TestAccuracy = svc.score(Xtest,Ytest)

print('='*40)
print('Support Vector Machines')
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
"""