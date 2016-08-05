import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random

iris_df = pd.read_csv("iris.data.csv",names=['sepal_length','sepal_width','petal_length','petal_width','class'])
iris_df['class'] = pd.Categorical(iris_df['class'])

"""Getting idea of the data"""
#%matplotlib inline
#iris_df.plot('sepal_length','sepal_width',kind='scatter')
#sns.lmplot('sepal_length', 'sepal_width', 
#           data=iris_df, 
#           fit_reg=True,
#           hue="class")
#plt.title('Scatter')
#plt.xlabel('Sepal Length')
#plt.ylabel('Sepal Width')

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
            x = float(iris_df['sepal_length'][[i]])
            y = float(iris_df['sepal_width'][[i]])
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