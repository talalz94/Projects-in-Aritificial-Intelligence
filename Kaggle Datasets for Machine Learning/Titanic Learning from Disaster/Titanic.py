import sys
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

sys.__stdout__ = sys.stdout

# Reading the values from data files
trainingData = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\train.csv''')
testData = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\test2.csv''')
testDataSolution = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\testResult.csv''')

totalData = [trainingData, testData]

# Dropping the Cabin, PassenergerId and Ticket attributes
trainingData = trainingData.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
testData = testData.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)

totalData = [trainingData, testData]

# Extracting the Titles from the name attribute
for values in totalData:
    values['Title'] = values.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Replacing rare titles with one title and fixing misspelled titles
for values in totalData:
    values['Title'] = values['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    
# Mapping the titles to an integer value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for values in totalData:
    values['Title'] = values['Title'].map(title_mapping)
    values['Title'] = values['Title'].fillna(0)

#Dropping the name attribute
trainingData = trainingData.drop(['Name'], axis=1)
testData = testData.drop(['Name'], axis=1)
totalData = [trainingData, testData]


# Replaceing Sex with 0 or 1
for values in totalData:
    values['Sex'] = values['Sex'].replace('female', 1)
    values['Sex'] = values['Sex'].replace('male', 0)
    

# Filling missing Age data with median Age of that Sex
for values in totalData:
    for i in range(0, 2):
        ageGuess = values[(values['Sex'] == i)]['Age'].dropna()
        ageGuess = ageGuess.median()

    for i in range(0, 2):   
        values.loc[ (values.Age.isnull()) & (values.Sex == i) ,'Age'] = ageGuess
        



#Categorizing Age according to range
for values in totalData:    
    values.loc[ values['Age'] <= 8, 'Age'] = 0
    values.loc[(values['Age'] > 8) & (values['Age'] <= 16), 'Age'] = 1
    values.loc[(values['Age'] > 16) & (values['Age'] <= 24), 'Age'] = 2
    values.loc[(values['Age'] > 24) & (values['Age'] <= 44), 'Age'] = 3
    values.loc[ values['Age'] > 44 & (values['Age'] <= 60), 'Age'] = 4
    values.loc[ values['Age'] > 60, 'Age'] = 5

totalData = [trainingData, testData]


#Replacing Parch and SibSp with Family Size
for values in totalData:
    values['FamilySize'] = 1 + values['SibSp'] + values['Parch']

trainingData = trainingData.drop(['Parch', 'SibSp', ], axis=1)
testData = testData.drop(['Parch', 'SibSp'], axis=1)
totalData = [trainingData, testData]

#Replacing missing Embarked values with the mode of the attribute

portMode = trainingData.Embarked.mode()[0]
for values in totalData:
    values['Embarked'] = values['Embarked'].fillna(portMode)


#Replacing values in Embarked with integers
for values in totalData:
    values['Embarked'] = values['Embarked'].replace('S', 0)
    values['Embarked'] = values['Embarked'].replace('C', 1)
    values['Embarked'] = values['Embarked'].replace('Q', 2)

#Categorizing Fare according to range

for values in totalData:
    values.loc[ values['Fare'] <= 8, 'Fare'] = 0
    values.loc[(values['Fare'] > 8) & (values['Fare'] <= 16), 'Fare'] = 1
    values.loc[(values['Fare'] > 16) & (values['Fare'] <= 32), 'Fare']   = 2
    values.loc[ values['Fare'] > 32, 'Fare'] = 3

totalData = [trainingData, testData]

#Preparing Data to be learned by Classifiers
Y_true = testDataSolution['Survived'].values.tolist()
X_train = trainingData.drop("Survived", axis=1)
Y_train = trainingData["Survived"]
X_test  = testData



#Support Vector Machine
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

svc_as = accuracy_score(Y_true, Y_pred)
svc_fl = f1_score(Y_true, Y_pred, average='binary')
svc_rs = recall_score(Y_true, Y_pred, average='binary')
svc_ps = precision_score(Y_true, Y_pred, average='binary')  

#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

lr_as = accuracy_score(Y_true, Y_pred)
lr_fl = f1_score(Y_true, Y_pred, average='binary')
lr_rs = recall_score(Y_true, Y_pred, average='binary')
lr_ps = precision_score(Y_true, Y_pred, average='binary')

#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)

dt_as = accuracy_score(Y_true, Y_pred)
dt_fl = f1_score(Y_true, Y_pred, average='binary')
dt_rs = recall_score(Y_true, Y_pred, average='binary')
dt_ps = precision_score(Y_true, Y_pred, average='binary')

#Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)

nb_as = accuracy_score(Y_true, Y_pred)
nb_fl = f1_score(Y_true, Y_pred, average='binary')
nb_rs = recall_score(Y_true, Y_pred, average='binary')
nb_ps = precision_score(Y_true, Y_pred, average='binary')

#Results
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression',  
              'Decision Tree', 'Naive Bayes'],
    'Accuracy' : [svc_as, lr_as, dt_as, nb_as],
    'Precision': [svc_ps, lr_ps, dt_ps, nb_ps],
    'Recall'   : [svc_rs, lr_rs, dt_rs, nb_rs],
    'F-measure': [svc_fl, lr_fl, dt_fl, nb_fl]})
models.sort_values(by='Model', ascending=False)
models = models[['Model','Accuracy', 'Precision', 'Recall', 'F-measure']]

print(models)
