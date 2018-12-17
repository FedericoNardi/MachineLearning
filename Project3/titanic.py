#%%
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

# Fill holes
TrainData.info()

# Most frequent port
freq_port = TrainData.Embarked.dropna().mode()[0]
print("The most frequent port is")
print(freq_port)



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
#g = sns.FacetGrid(TrainData, col='Survived')
#g.map(plt.hist, 'Age', bins=20)

#sns.factorplot(y="Age",x="Pclass", data=TrainData,kind="box")
#sns.factorplot(y="Age",x="SibSp", data=TrainData,kind="box")
# Fill holes in Age
NaNIndex = list(TrainData['Age'][TrainData['Age'].isnull()].index)

for i in range(len(NaNIndex)):
	AgeMedian = TrainData['Age'].mean()
	AgePredict = TrainData["Age"][ ((TrainData['Parch'] == TrainData.iloc[i]["Parch"]) & (TrainData['Pclass'] == TrainData.iloc[i]["Pclass"]) ) ] .median()
	if not np.isnan(AgePredict)==True:
		TrainData['Age'][NaNIndex[i]] = AgePredict
	else:
		TrainData['Age'][NaNIndex[i]] = AgeMedian
#%%
# Drop useless

# Create Family size variable
Family = TrainData['SibSp']+TrainData['Parch']
TrainData['Family'] = Family

#Drop not relevant data
TrainData.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)

TrainData.info()

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

#%%
#print(TrainData['Embarked'].head())

# SETTING UP REGRESSIONS
