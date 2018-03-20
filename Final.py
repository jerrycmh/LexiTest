import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

import scipy.io as sio
import matplotlib.pyplot as plt
#import seaborn as sns
#%config InlineBackend.figure_format = 'retina'

letterData = []
with open('letter-recognition.data.txt') as infile:
    line = infile.readline()
    while(line):
        letterData.append(line.strip('\n').split(','))
        line = infile.readline()

unique = set()
p_LetterData = []
for d in letterData:
    tmp = []
    if(ord(d[0]) <= 77 and ord(d[0]) >= 65):
        tmp.append(1)
        tmp.extend(d[1:])
    elif(ord(d[0]) > 77 and ord(d[0]) <= 90):
        tmp.append(0)
        tmp.extend(d[1:])
    tmp = [ int(x) for x in tmp ]
    p_LetterData.append(tmp)

print(p_LetterData[0:5])

np.random.shuffle(p_LetterData)    # Shuffle the data.

p_LetterData = np.array(p_LetterData)
data_Y = p_LetterData[:,0]
data_X = p_LetterData[:, 1:len(p_LetterData[0])]
print(data_X.shape)
print(data_Y.shape)

ratio = 0.2
X_test, X_train, y_test, y_train = train_test_split(data_X, data_Y, test_size=ratio, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(X_train[0:10])

def get_gamma(width):
    return 1/(2*(width**2))

width = [0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1,2]
gamma_List = [get_gamma(x) for x in width]
print(gamma_List)
print("finish gamma")

"""
width = [0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1,2]
gamma_List = [get_gamma(x) for x in width]
#C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4)]
C_list = [10**(-3), 10**(-2), 10**(-1), 1, 10] # Different C to try.
#C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10, 10**(2), 10**(3)] # Different C to try.
parameters = {'kernel':('linear','rbf'), 'C': C_list, 'gamma': gamma_List}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)


C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4)]

print("===============")

parameters = {'C': C_list}
svc = svm.SVC(kernel='poly')
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)
"""
C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4),10**(-3), 10**(-2), 10**(-1),1,10]
parameters = {'kernel':('linear','rbf'), 'C': C_list, 'gamma': gamma_List}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)

svc = svm.SVC(kernel = clf.best_params_['kernel'], C = clf.best_params_['C'], gamma = clf.best_params_['gamma'])
svc.fit(X_train, y_train)
print(svc.score(data_X, data_Y))