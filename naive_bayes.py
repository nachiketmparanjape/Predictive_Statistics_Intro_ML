import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.naive_bayes import GaussianNB

#Loading weight data to the dataframe
weight_df = pd.read_csv("ideal_weight.csv")

#cleaning
weight_df = weight_df.rename(index=str, columns={"'id'": "id", "'sex'": "sex","'actual'": "actual","'ideal'": "ideal","'diff'": "diff"})
weight_df["sex"] = map(lambda x: str(x[1:-1]), weight_df["sex"])
weight_df["sex"] = weight_df['sex'].astype('category')

#Visualizations
#sns.distplot(weight_df["actual"])
#sns.distplot(weight_df["ideal"])
#sns.distplot(weight_df["diff"])
#sns.countplot(x="sex", data=weight_df)

#Creating arrays X and Y to feed to the naive bayes function (Syntactical adjustments)

""" X """
df = pd.DataFrame()
df['actual']=weight_df['actual']
df['ideal']=weight_df['ideal']
df['diff']=weight_df['diff']

x1 = np.array(df.iloc[[0]])
x2 = np.array(df.iloc[[1]])
Xfull = np.concatenate((x1,x2),axis=0)

for i in range(2,len(df)):
    Xfull = np.concatenate((Xfull,np.array(df.iloc[[i]])),axis=0)
    
X = Xfull[57:] #adjust the range here to change training and testing datasets

""" Y """
Yfull = weight_df['sex']
Y = Yfull[57:] #adjust the range here to change training and testing datasets

#GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

#Calculating accuracy
weight_df['psex'] = ""
for i in range(len(weight_df)):
    psex = clf.predict([Xfull[i]])
    weight_df['psex'][[i]] = psex


#Calculating accuracy
pred = pd.Series(weight_df['sex']==weight_df['psex'])
test_data = pred[:57] #adjust the range here to change training and testing datasets

total = test_data.count()
True_Positives = test_data[test_data==True].count()

accuracy = round((100 - (float(total - True_Positives)/total)*100), 2)
print ( "The accuracy on the test dataset is " + str(accuracy) + " %")