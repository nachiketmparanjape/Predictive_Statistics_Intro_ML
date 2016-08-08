import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

undf = pd.read_csv('un.csv')
undf['country'] = pd.Categorical(undf['country'])
undf['region'] = pd.Categorical(undf['region'])
kmeans_df = undf[['lifeMale','lifeFemale','infantMortality','GDPperCapita']].dropna()

krange = range(1,11)
avg_sum_of_squares_list = []

for k in krange:
    whitened = scipy.cluster.vq.whiten(kmeans_df)
    centroid = scipy.cluster.vq.kmeans(whitened,k)[0]
    #    distances = pd.Series()
    
     # Contains indices for a point and corrensponding closest centroid
    coordinates = pd.Series(index=[kmeans_df.index])
        
    # Determine closest centroid for each point
    minimum_dist_list = []
    for i in kmeans_df.index:
        current_point = kmeans_df.ix[i]
        distlist = []
        
        for c in centroid:
            cx = c[0]
            cy = c[1]
            cz = c[2]
            cw = c[3]
            x = current_point[0]
            y = current_point[1]
            z = current_point[2]
            w = current_point[3]
            #print x
            #print cx
            distance = np.sqrt((cx-x)**2 + (cy-y)**2 + (cz-z)**2 + (cw-w)**2)
            distlist.append(distance)
        
        #min_dist_index = distlist.idxmin()
        
        #coordinates[i] = min_dist_index
        #print distlist
        #print min(distlist)
        minimum_dist_list.append(min(distlist))
    
    
    avg_sum_of_squares_list.append(reduce(lambda x, y: x + y, minimum_dist_list) / len(minimum_dist_list))
        
#Create a dataframe for plotting
avg_distances_df = pd.DataFrame({'k':krange,'avg_min_distance':avg_sum_of_squares_list})
avg_distances_df.plot(x='k',y='avg_min_distance')
        
        
        