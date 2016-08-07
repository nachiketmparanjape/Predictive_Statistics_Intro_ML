import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #in case needed
from sklearn.cross_validation import KFold
from sklearn import linear_model

loansData_raw = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')
loansData_raw['index'] = range(len(loansData_raw))
loansData = loansData_raw.set_index('index',drop=True)

loansData['FICO.Score'] = map(lambda x: int(x[0:3]), loansData['FICO.Range'])
loansData['Interest.Rate'] = map(lambda x: float(x[0:-1]), loansData['Interest.Rate'])
loansData['Loan.Length'] = map(lambda x: int(x[0:-7]), loansData['Loan.Length'])
loansData['Debt.To.Income.Ratio'] = map(lambda x: float(x[0:-1]), loansData['Debt.To.Income.Ratio'])

#loansData.to_csv('loansData_clean.csv', header=True, index=False)
#plt.figure()
#p = loansData['FICO.Score'].hist()
#plt.show()

#Plotting Histograms instead of the meaningless plot of the variables plotted against themselves
#a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')

folds = KFold(len(loansData),n_folds=500)

#testdf = pd.DataFrame(loansData['Interest.Rate','Amount.Requested','FICO.Range'])
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']
    

#
#
#3 variable linear regression
#count = 0
scores = []
for train, test in folds:
    #count += 1
    #print train[:5]
    #print test[:5]
    
    #independent variable
    y = np.array(intrate[train]).transpose()

    #Dependent variables
    x1 = np.array(fico[train]).transpose()
    x2 = np.array(loanamt[train]).transpose()
    X = np.column_stack([x1,x2])
    
    #independent variable
    ytest = np.array(intrate[test]).transpose()

    #Dependent variables
    x1test = np.array(fico[test]).transpose()
    x2test = np.array(loanamt[test]).transpose()
    Xtest = np.column_stack([x1test,x2test])

    model = linear_model.LinearRegression(fit_intercept = True)
    f = model.fit(X,y)
    scores.append(model.score(Xtest,ytest))
    
print (reduce(lambda x, y: x + y, scores) / len(scores))

    
    
    
    