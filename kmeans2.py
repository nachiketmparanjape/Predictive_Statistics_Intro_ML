import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

undf = pd.read_csv('un.csv')
undf['country'] = pd.Categorical(undf['country'])
undf['region'] = pd.Categorical(undf['region'])
kmeans_df = undf[['lifeMale','lifeFemale','infantMortality','GDPperCapita']].dropna()

k = 3 #Decided from kmeans1

whitened = scipy.cluster.vq.whiten(kmeans_df)
centroid = scipy.cluster.vq.kmeans(whitened,k)[0]
#    distances = pd.Series()

 # Contains indices for a point and corrensponding closest centroid
coordinates = pd.Series(index=[kmeans_df.index])
    
# Determine closest centroid for each point
minimum_dist_list = []
for i in kmeans_df.index:
    current_point = kmeans_df.ix[i]
    distlist = pd.Series(index = [0,1,2])
    counter = 0
    
    for c in centroid:
        cx = c[3]
        #cy = c[1]
        #cz = c[2]
        
        x = current_point[3]
        #y = current_point[1]
        #z = current_point[2]
        
        #print x
        #print cx
        distance = (cx-x)
        distlist[counter] = distance
        counter += 1
    
    #min_dist_index = distlist.idxmin()
    
    #coordinates[i] = min_dist_index
    #minimum_dist_list.append(distlist.idxmin())
    minimum_dist_list.append(distlist.idxmin())
kmeans_df['marker'] = pd.Categorical(minimum_dist_list)

sns.lmplot('GDPperCapita', 'infantMortality', 
           data=kmeans_df, 
           fit_reg=False,
           hue="marker")
           
sns.lmplot('GDPperCapita', 'lifeMale', 
           data=kmeans_df, 
           fit_reg=False,
           hue="marker")
           
sns.lmplot('GDPperCapita', 'lifeFemale', 
           data=kmeans_df, 
           fit_reg=False,
           hue="marker")