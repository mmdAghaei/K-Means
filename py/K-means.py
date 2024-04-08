# Import Package
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Make Data
x,y = make_blobs(n_samples=400,n_features=2,centers=3)

# Show Data
plt.scatter(x[:,0],x[:,1],c="blue")
plt.show()

# K-Means
model = KMeans(n_clusters=3)
model.fit(x)

# Set Centers and labels
centers = model.cluster_centers_
labels = model.labels_

# Show Result
plt.scatter(x[:,0],x[:,1],c=labels)
plt.scatter(centers[:,0],centers[:,1],c="Red",marker="x")
plt.show()