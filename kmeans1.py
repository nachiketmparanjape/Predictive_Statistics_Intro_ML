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
for k in krange:
    whitened = scipy.cluster.vq.whiten(kmeans_df)
    centroid = scipy.cluster.vq.kmeans(whitened,k)[0]
    distances = pd.Series()
    
    # Determine closest centroid for each point
    for i in kmeans_df.index:
        current_point = kmeans_df.ix[i]
        distlist = pd.Series()
        j = 0
        for c in centroid:
            cx = float(c[0])
            cy = float(c[1])
            cz = float(c[2])
            cw = float(c[3])
            x = float(current_point[0])
            y = float(current_point[1])
            z = float(current_point[2])
            w = float(current_point[3])
            distance = np.sqrt((cx-x)**2 + (cy-y)**2 + (cz-z)**2 + (cw-w)**2)
            distlist[[j]] = distance
            j += 1
        #avg_dist = reduce(lambda x, y: x + y, distlist)
        
        
        