import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

# Set seed for reproducible results
np.random.seed(414)

# Gen toy data
X = np.linspace(0, 15, 1000)
y = 3 * np.sin(X) + np.random.normal(1 + X, .2, 1000)

train_X, train_y = X[:700], y[:700]
test_X, test_y = X[700:], y[700:]

train_df = pd.DataFrame({'X': train_X, 'y': train_y})
test_df = pd.DataFrame({'X': test_X, 'y': test_y})

# Linear Fit
poly_1 = smf.ols(formula='y ~ 1 + X', data=train_df).fit()

# Quadratic Fit
poly_2 = smf.ols(formula='y ~ 1 + X + I(X**2)', data=train_df).fit()

print ("Linear fit rsquared - " + str(poly_1.rsquared))
print ("2nd degree fit rsquared - " + str(poly_2.rsquared))