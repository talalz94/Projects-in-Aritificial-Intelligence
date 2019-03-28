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
from collections import Counter

sys.__stdout__ = sys.stdout

# Reading the values from data files
trainingData = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\spam.csv''', encoding='latin-1')
testData = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\spamTest.csv''', encoding='latin-1')
testDataSolution = pd.read_csv(r'''C:\Users\bilal\Desktop\talal\6TH SEMESTER\AI\smsSolution.csv''')

# Removing unneccessary columns
trainingData = trainingData.drop([trainingData.columns[2], trainingData.columns[3], trainingData.columns[4]], axis=1)
testData = testData.drop([testData.columns[2], testData.columns[3], testData.columns[1]], axis=1)

totalData = [trainingData, testData]

# Converting v1 attribute to 0,1 instead of ham or spam
v1map = {'ham': 0, 'spam': 1}
trainingData['v1'] = trainingData.replace({'v1': v1map})
testDataSolution['v1'] = testDataSolution.replace({'v1': v1map})
totalData = [trainingData, testData]


#Creating new attribute Length (of the SMS)
for values in totalData:
    values['SMSLength'] = values['v2'].str.len()

#Dividing the SMS Length into 3 categories
for values in totalData:
    values.loc[ values['SMSLength'] <= 15, 'SMSLength'] = 0
    values.loc[(values['SMSLength'] > 15) & (values['SMSLength'] <= 250), 'SMSLength'] = 1
    values.loc[ values['SMSLength'] > 300, 'SMSLength'] = 2

#100 most frequent occuring unique words in spam SMS

df = trainingData.loc[trainingData['v1'] == 1]
spamWordFreq = Counter(" ".join(df["v2"]).split()).most_common(100)
cat = Counter(" ".join(df["v2"]).split()).most_common(100)
df = trainingData.loc[trainingData['v1'] == 0]
hamWordFreq = Counter(" ".join(df["v2"]).split()).most_common(100)

hamWordFreq = [i[0] for i in hamWordFreq]
spamWordFreq = [i[0] for i in spamWordFreq]

spamWords = set(spamWordFreq) - set(hamWordFreq)

#Creating new attribute SpamWordFreq
spamWordFreq = []
for values in trainingData['v2']:
    sms = values
    smsfreq = 0
    for word in spamWords:
        if word in sms.split():
            smsfreq = smsfreq + 1
    spamWordFreq.append(smsfreq)
trainingData['spamWordFreq'] = spamWordFreq

spamWordFreq = []
for values in testData['v2']:
    sms = values
    smsfreq = 0
    for word in spamWords:
        if word in sms.split():
            smsfreq = smsfreq + 1
    spamWordFreq.append(smsfreq)
testData['spamWordFreq'] = spamWordFreq

totalData = [trainingData, testData]

#Dividing the SpamWordFreq into 3 categories

for values in totalData:
    values.loc[ values['spamWordFreq'] < 1, 'spamWordFreq'] = 0
    values.loc[ values['spamWordFreq'] == 1, 'spamWordFreq'] = 1
    values.loc[(values['spamWordFreq'] > 1) & (values['spamWordFreq'] < 3), 'spamWordFreq'] = 2
    values.loc[ values['spamWordFreq'] > 3, 'spamWordFreq'] = 3    
    
#Dropping the SMS Attribute
trainingData = trainingData.drop(['v2'], axis=1)
testData = testData.drop(['v2'], axis=1)
totalData = [trainingData, testData]

#Preparing Data to be learned by Classifiers
Y_true = testDataSolution['v1'].values.tolist()
X_train = trainingData.drop("v1", axis=1)
Y_train = trainingData["v1"]
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

