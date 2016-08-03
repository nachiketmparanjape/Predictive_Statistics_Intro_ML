import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #in case needed
from sklearn.cross_validation import KFold
from sklearn import linear_model

loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')


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

folds = KFold(len(loansData),n_folds=10)

#testdf = pd.DataFrame(loansData['Interest.Rate','Amount.Requested','FICO.Range'])
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']
#
#
#3 variable linear regression
for train, test in folds:
    
    #independent variable
    y = np.matrix(intrate[train]).transpose()

    #Dependent variables
    #X = np.matrix(fico[train]).transpose()
    X = np.matrix(loanamt[train]).transpose()

    #X = np.column_stack([x1,x2])

    model = linear_model.LinearRegression(fit_intercept = True)
    f = model.fit(X,y)
    