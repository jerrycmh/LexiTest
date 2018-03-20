import numpy as np
import gzip
#from statistics import mode
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

data = []
with gzip.open('covtype.data.gz','rb') as infile:
    for line in infile:
        data.append( line.decode("utf-8").strip('\n').split(','))
print(len(data))

Data = []
for d in data:
    d = [ int(x) for x in d ]
    Data.append(d)


np.random.shuffle(Data)    # Shuffle the data.

Data = np.array(Data)
data_Y = Data[:,-1]
data_X = Data[:, 1:len(Data[0])]
print(data_X.shape)
print(data_Y.shape)

tmp = []
for idx in data_Y:
    if data_Y[idx] == 2:
        tmp.append(1)
    else:
        tmp.append(0)
data_Y = tmp
data_Y = np.array(data_Y)
print(data_X.shape)
print(data_Y.shape)

ratio = 0.2
X_test, X_train, y_test, y_train = train_test_split(data_X[:20000], data_Y[:20000], test_size=ratio, random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def get_gamma(width):
    return 1/(2*(width**2))

width = [0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1,2]
gamma_List = [get_gamma(x) for x in width]
print(gamma_List)

print("finish gamma")

width = [0.001, 0.005, 0.01, 0.05, 0.1,0.5, 1,2]
gamma_List = [get_gamma(x) for x in width]
#C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4)]
C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4),10**(-3), 10**(-2), 10**(-1), 1,10] # Different C to try.
#C_list = [1, 10]
#C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4)]
#C_list = [10**(-3), 10**(-2), 10**(-1), 1]
# [10, 10**(2), 10**(3)] # Different C to try.
parameters = {'kernel':('linear','rbf'), 'C': C_list, 'gamma': gamma_List}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)

svc = svm.SVC(kernel = clf.best_params_['kernel'], C = clf.best_params_['C'], gamma = clf.best_params_['gamma'])
svc.fit(X_train, y_train)
print(svc.score(data_X, data_Y))





