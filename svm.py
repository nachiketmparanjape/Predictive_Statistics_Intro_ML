from sklearn import datasets
import numpy as np
iris = datasets.load_iris()

#Adapted from https://github.com/jakevdp/sklearn_scipy2013
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def plot_estimator(estimator, X, y):
    estimator.fit(X, y)
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.axis('tight')
    plt.axis('off')
    plt.tight_layout()
    
    
import matplotlib.pyplot as plt

plt.scatter(iris.data[:, 1], iris.data[:, 2], c=iris.target)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

#Fit SVC from scikit learn
from sklearn import svm
svc = svm.SVC(kernel='linear')
from sklearn import datasets
X = iris.data[0:100, [1,2]]
y = iris.target[0:100]
svc.fit(X, y)

plot_estimator(svc,X,y)