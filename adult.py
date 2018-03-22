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
from sklearn.preprocessing import OneHotEncoder

data = []
with open('adult.data') as infile:
    for line in infile:
        data.append( line.strip('\n').split(','))
    print(len(data))

data = np.array(data)
print(data.shape)
print(data[0:5])

columns = ['age', 'workclass', 'fnlwgt','education','education-num', 'marital-status',
           'occupation','relationship','race','sex','capital-gain','capital-loss',
           'hours-per-week','native-country', 'label']
dct = {}
for row in data:
    count = 0
    for i in range(len(row)):
        if columns[i] in dct.keys():
            dct[columns[i]].append(row[i])
        else:
            dct[columns[i]] = [row[i]]

workList = list(set(dct['workclass']))
#print(workList)
eduList = list(set(dct['education']))
#print(eduList)
marryList = list(set(dct['marital-status']))
#print(marryList)
occuList = list(set(dct['occupation']))
#print(occuList)
relationList = list(set(dct['relationship']))
#print(relationList)
raceList = list(set(dct['race']))
#print(raceList)
sexList = list(set(dct['sex']))
#print(sexList)
countryList = list(set(dct['native-country']))
#print(countryList)

Data = range(488430)
Data = np.reshape(Data, (32562,15))


r = 0
for row in data:
    #tmp = []
    for i in range(len(row)):
        # age         
        if i == 0:
            if row[i] == '':
                Data[r][i] = 0
                #tmp.append(0)
            else:
                Data[r][i] = int(row[i])
#               tmp.append(int(row[i]))
        # work class
        if i == 1:
            Data[r][i] = workList.index(row[i])
            #tmp.append(workList.index(row[i])) 
        # fnlwgt
        if i == 2:
            if row[i] == '':
                Data[r][i] = 0
#tmp.append(0)
            else:
                Data[r][i] = int(row[i])
#                tmp.append(int(row[i]))
        # education 
        if i == 3:
            Data[r][i] = eduList.index(row[i])
            #tmp.append(eduList.index(row[i])) 
        # education Num
        if i == 4:
            if row[i] == '':
                Data[r][i] = 0
                #tmp.append(0)
            else:
                Data[r][i] = int(row[i])
                #tmp.append(int(row[i]))
        # marital status
        if i == 5:
            Data[r][i] = marryList.index(row[i])
            #tmp.append(marryList.index(row[i])) 
        # occupation:
        if i == 6:
            Data[r][i] = occuList.index(row[i])
            #tmp.append(occuList.index(row[i]))
        # relationship
        if i == 7:
            Data[r][i] = relationList.index(row[i])
            #tmp.append(relationList.index(row[i]))
        # race
        if i == 8:
            Data[r][i] = raceList.index(row[i])
            #tmp.append(raceList.index(row[i]))
        # sex
        if i == 9:
            Data[r][i] = sexList.index(row[i])
#            tmp.append(sexList.index(row[i]))
        # capital gain
        if i == 10:
            if row[i] == '':
                Data[r][i] = 0
#                tmp.append(0)
            else:
                Data[r][i] = int(row[i])
                #tmp.append(int(row[i]))
        # capital loss
        if i == 11:
            if row[i] == '':
                Data[r][i] = 0
                #tmp.append(0)
            else:
                Data[r][i] = int(row[i])
#                tmp.append(int(row[i]))
        # hours per week 
        if i == 12:
            if row[i] == '':
                Data[r][i] = 0
                #tmp.append(0)
            else:
                Data[r][i] = int(row[i])
#                tmp.append(int(row[i]))
        # native country
        if i == 13:
            Data[r][i] = countryList.index(row[i])
            #tmp.append(countryList.index(row[i]))
        # label 
        if i == 14:
            if row[i] == ' <=50K':
                Data[r][i] = 0 
#                tmp.append(0)
            else:
                Data[r][i] = 1 
                #tmp.append(1) 
    r+= 1
    

np.random.shuffle(Data) 
data_Y = Data[:,-1]
data_X = Data[:, 1:len(Data[0])]
data_Y = np.array(data_Y)


ratio = 0.05
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
C_list = [10**(-7), 10**(-6), 10**(-5)]
#C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4),10**(-3)] # Different C to try.
#C_list = [1, 10]
#C_list = [10**(-7), 10**(-6), 10**(-5),10**(-4)]
#C_list = [10**(-3), 10**(-2), 10**(-1), 1]
# [10, 10**(2), 10**(3)] # Different C to try.
'''
parameters = {'kernel':('linear','rbf'), 'C': C_list, 'gamma': gamma_List}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(X_train, y_train)

print(clf.best_params_)
'''

svc = svm.SVC(kernel = 'poly', C = 10**(-7))
svc.fit(X_train, y_train)
print(svc.score(X_train, y_train))
print(svc.score(data_X, data_Y))






















labelList = list(set(dct['label']))