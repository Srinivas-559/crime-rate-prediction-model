import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt
data = pd.read_csv("/Users/srinivasgollapalli/Desktop/crime.csv")
data
data.Magnitude.unique()
arr = data.Magnitude.unique()
arr = np.delete(arr, 7)
arr = arr.astype(int)
avg = np.average(arr)
avg
data.Magnitude = data.Magnitude.replace(to_replace ="ARSON",value ="9")
data.Magnitude.unique()
data['Magnitude'] = data['Magnitude'].astype(int)
data.Magnitude.unique()
X = data[['Latitude', 'Longitude', 'Magnitude']]
X = np.array(X)
X
plt.scatter(data.Latitude,data.Longitude)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
import numpy as np
from fcmeans import FCM
my_model = FCM(n_clusters=3,random_state=42)
my_model.fit(X)
centers = my_model.centers
fcm_clusters = my_model.predict(X)
fcm_clusters
plt.scatter(data.Latitude,data.Longitude,s=None, c=fcm_clusters, cmap=None )
import statsmodels.api as sm
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
kmeans = KMeans(3,random_state=42,max_iter=20,verbose=10)
kmeans.fit(X)
kmeans_clusters = kmeans.fit_predict(X)
kmeans_clusters
len(kmeans_clusters)
cnt_match = 0
for i in range(0,33116):
    if fcm_clusters[i] == kmeans_clusters[i]:
        cnt_match = cnt_match + 1

res= cnt_match/len(kmeans_clusters)
print("Accuracy increased by :",res)
latit = float(input("Enter Latitude = "))
longit = float(input("Enter Longitude = "))
rate=0.0
c=0
for i in range(0,len(X)):
    if (X[i,0] == latit) & (X[i,1] == longit):
        rate = X[i,2]
        c+=1
print("This location is",100-c,"% safe" )
        
        


