import numpy as np
import pandas as pd
import random
from sklearn import decomposition
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

iris_df = pd.read_csv("iris.data.csv",names=['sepal_length','sepal_width','petal_length','petal_width','class'])
iris_df['class'] = pd.Categorical(iris_df['class'])

np.random.seed(5)

#iris = datasets.load_iris()
X = iris_df[['sepal_length','sepal_width','petal_length','petal_width']]
y = iris_df['class']


"""Apply PCA to X"""
pca = decomposition.PCA(n_components=3)
pca.fit(X)
Xtarray = pca.transform(X)
Xt = pd.DataFrame(Xtarray)

"""Apply LDA to X,y"""
lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)
X_r2 = pd.DataFrame(X_r2)


"""Nearest Neighbours"""

"""Predicting class from the sepal width and length using knn"""
def pick_new_point():
#Pick a new point, programmatically at random    
    s_width = random.uniform(min(iris_df['sepal_width']),max(iris_df['sepal_width']))
    s_length = random.uniform(min(iris_df['sepal_length']),max(iris_df['sepal_length']))
    return s_length, s_width
    
def knn(klist):
    majority_classes = []
    for j in klist:
        random_point = pick_new_point()
        dist_list = []
        for i in range(len(iris_df)):
            newx = float(random_point[0])
            newy = float(random_point[1])
            x = float(X['sepal_length'][[i]])
            y = float(X['sepal_width'][[i]])
            distance = np.sqrt((newx-x)**2 + (newy-y)**2) #Euclidian distance of every point
            dist_list.append(distance)
        iris_df['distance'] = dist_list
        # iris_df.tail()

        k_df = iris_df.sort(columns = 'distance')[:j]
        #k_df

        # this is to return the majority class from a given Series of categorical data
        majority_class = k_df['class'].value_counts().index[0]
        majority_classes.append(majority_class)
    k_values_df = pd.DataFrame({'k':klist,'majority_class':majority_classes})
    return k_values_df
    
""" Nearest Neighbours with LDA """

def knn_lda(klist):
    majority_classes = []
    for j in klist:
        random_point = pick_new_point()
        dist_list = []
        for i in range(len(iris_df)):
            newx = float(random_point[0])
            newy = float(random_point[1])
            x = float(X_r2[0][[i]])
            y = float(X_r2[1][[i]])
            distance = np.sqrt((newx-x)**2 + (newy-y)**2) #Euclidian distance of every point
            dist_list.append(distance)
        iris_df['distance'] = dist_list
        # iris_df.tail()

        k_df = iris_df.sort(columns = 'distance')[:j]
        #k_df

        # this is to return the majority class from a given Series of categorical data
        majority_class = k_df['class'].value_counts().index[0]
        majority_classes.append(majority_class)
    k_values_df = pd.DataFrame({'k':klist,'majority_class':majority_classes})
    return k_values_df
        
#print knn(range(10))
#print knn_with_pca(range(10))