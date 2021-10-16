import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
print(data.columns.values[3])
X = data.iloc[:,[1,4]].values
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
try:
    y_kmeans = kmeans.fit_predict(X)
except ValueError:
    print("hello")
color = ['red', 'blue', 'green', 'cyan', 'magenta','orange', 'purple', 'gray']
for i in range(0, 5):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=color[i], label='Cluster %d' % (i + 1))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black',
            marker="*", label='Centroids')
plt.legend()
plt.show()

