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
from sklearn.utils import shuffle
from sklearn import preprocessing

sys.__stdout__ = sys.stdout

# Reading the values from data files
trainingData = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\voice2.csv''')
testData = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\voiceTest.csv''')
testDataSolution = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\voiceSolution.csv''')

#Converting label dataType to string from object
trainingData['label'] = trainingData['label'].astype('str') 
testDataSolution['label'] = testDataSolution['label'].astype('str') 

# Converting label attribute to 0,1 instead of male, female
v1map = {'male': 0, 'female': 1}
trainingData['label'] = trainingData['label'].map(v1map)
testDataSolution['label'] = testDataSolution['label'].map(v1map)

totalData = [trainingData, testData]

#Dropping less important attributes (through found through describe())
trainingData = trainingData.drop(['modindx', 'mindom','maxfun','minfun','sp.ent','Q75','Q25'], axis=1)
testData = testData.drop(['modindx', 'mindom','maxfun','minfun','sp.ent','Q75','Q25'], axis=1)

totalData = [trainingData, testData]

#Normalizing the attributes
x = trainingData.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
trainingData=pd.DataFrame(x_scaled, columns=trainingData.columns)

#Rounding to 2 decimal places
trainingData = trainingData.round(2)
trainingData['label'] = trainingData['label'].astype('int') 
testDataSolution['label'] = testDataSolution['label'].astype('int')
#trainingData = trainingData.astype('int')
totalData = [trainingData, testData]


#Preparing Data to be learned by Classifiers
Y_true = testDataSolution['label'].values.tolist()
X_train = trainingData.drop("label", axis=1)
Y_train = trainingData["label"]
Y_train = Y_train.astype('int')
X_test  = testData

###Support Vector Machine
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

