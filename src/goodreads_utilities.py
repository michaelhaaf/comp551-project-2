#good reads 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv
path="br.csv"





nRowsRead = 312073 # specify 'None' if want to read whole file
# br.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv(path, delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'br.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


df1=df1.drop('bookID',axis=1)
df1=df1.drop('title',axis=1)
df1=df1.drop('author',axis=1)
df1=df1.drop('rating',axis=1)
df1=df1.drop('ratingsCount',axis=1)
df1=df1.drop('reviewsCount',axis=1)
df1=df1.drop('reviewerName',axis=1)



df1=df1.dropna()

df1=df1[df1.reviewerRatings != 3 ]

df1.reviewerRatings[df1.reviewerRatings <=2] = 0 

df1.reviewerRatings[df1.reviewerRatings >=4] = 1


stats= df1['reviewerRatings'].value_counts()

stats




df_00s=df1[df1.reviewerRatings == 0 ]
df_11s=df1[df1.reviewerRatings == 1 ]

################################################################################################
    
def rand(r): #chooses reviews randomly to create a balances set using random state r
    randomsample_1= df_11s.sample(5712,random_state=r)
    
    #randomsample_1= df_11s[0:5172]
    
    frames= [df_00s,randomsample_1]
    
    d=pd.concat(frames)
    
    d=d.values
    X_goodreads=[]
    y_goodreads=[]
    
    
    
    for i in range (len(d)):
        X_goodreads.append(d[i][1])
        y_goodreads.append(d[i][0])
        
    return X_goodreads,y_goodreads
    




import numpy as np
def random_subset(X_goodreads,y_goodreads,RS_seed,size_of_RS):
    np.random.seed(seed=RS_seed)
    idx = np.random.choice(np.arange(len(X_goodreads)), size_of_RS, replace=False)
    
    X_goodreads_new=[]
    y_goodreads_new=[]
    for i in range (len(idx)):
        
        X_goodreads_new.append(X_goodreads[idx[i]])
        y_goodreads_new.append(y_goodreads[idx[i]])
    
    return X_goodreads_new,y_goodreads_new




