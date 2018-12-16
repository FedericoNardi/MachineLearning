import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load TrainData
TrainData = pd.read_csv('Dataset/train.csv')
TestData = pd.read_csv('Dataset/test.csv')

"""
g = sns.FacetGrid(TrainData,col='Parch')
g.map(sns.distplot, 'Age', bins=20, norm_hist=True)
plt.show()
"""

--------------------------------------------------------------------
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


----------------------------------------------------------------------
# Drop useless

# Create Family size variable
Family = TrainData['SibSp']+TrainData['Parch']
TrainData['Family'] = Family

#Drop not relevant data
TrainData.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'],axis=1,inplace=True)

TrainData.info()


---------------------------------------------------------------------
# Convert categorical to int

TrainData['Embarked'] = TrainData['Embarked'].map( {'S':0, 'C':1, 'Q':2} ).astype(np.int64)
TrainData['Sex'] = TrainData['Sex'].map( {'male':0,'female':1} )
#print(TrainData['Embarked'].head())

---------------------------------------------------------------------
# SETTING UP REGRESSIONS
