import pandas as pd
from sklearn import datasets
from sklearn import cross_validation
from sklearn import svm
import numpy as np

#loading iris data in the dataframe
iris = datasets.load_iris()
X = iris.data
y = iris.target
total = pd.DataFrame(Y)
total['target'] = x

"""Using Cross-Validation"""
folds = cross_validation.KFold(len(total),n_folds=5)

#Function that performs SVC on each set and stores each score to a list

scores = []
for train, test in folds:
    #count += 1
    #print len(train)
    #print len(test)
    
    
    #independent variable
    y = np.array(total['target'][train])

    #Dependent variables
    x1 = np.array(total[0][train]).transpose()
    x2 = np.array(total[1][train]).transpose()
    x3 = np.array(total[2][train]).transpose()
    x4 = np.array(total[3][train]).transpose()
    X  = np.column_stack([x1,x2,x3,x4])
    
    #independent variable
    ytest = np.array(total['target'][test])

    #Dependent variables
    x1test = np.array(total[0][test]).transpose()
    x2test = np.array(total[1][test]).transpose()
    x3test = np.array(total[2][test]).transpose()
    x4test = np.array(total[3][test]).transpose()
    Xtest  = np.column_stack([x1test,x2test,x3test,x4test])

    svc = svm.SVC(kernel='linear')
    svc.fit(X,y)
    scores.append(svc.score(Xtest,ytest))
    
#Average of the scores
print sum(scores)/len(scores)