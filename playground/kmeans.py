# imports 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

path_image = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png'
path_image = 'images/landscape.jpeg'

plt.rcParams["figure.figsize"] = (12, 50) 

# load image 
img = cv.imread(path_image)
Z = img.reshape((-1, 3))  # Flatten the image to a list of pixels

# Convert to np.float32
Z = np.float32(Z)

# Set the number of clusters (K)
K = 4
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(Z)  # Cluster each pixel

# Get the color centers for each cluster
centers = np.uint8(kmeans.cluster_centers_)

# Recolor each pixel based on its cluster label
res = centers[labels]
res2 = res.reshape(img.shape)  # Reshape to the original image dimensions

# Display the clustered result
cv.imshow('KMeans Clustered Image', res2)
cv.waitKey(0)
cv.destroyAllWindows()
