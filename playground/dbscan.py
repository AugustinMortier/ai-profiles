# imports 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

path_image = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png'
#path_image = 'images/landscape.jpeg'

plt.rcParams["figure.figsize"] = (12, 50) 

# load image
img = cv.imread(path_image)
Z = img.reshape((-1, 3))  # Flatten image to list of pixels

# Convert to np.float32 and standardize for DBSCAN
Z = np.float32(Z)
Z_scaled = StandardScaler().fit_transform(Z)  # Scale features to have zero mean and unit variance

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=10)  # Adjust eps and min_samples based on your data
labels = dbscan.fit_predict(Z_scaled)

# Find unique labels (clusters) and assign random colors to each
unique_labels = np.unique(labels)
colors = np.random.randint(0, 255, (len(unique_labels), 3))

# Create an output image where each pixel's color corresponds to its cluster label
res = np.zeros_like(Z, dtype=np.uint8)
for label in unique_labels:
    if label == -1:  # DBSCAN uses -1 for noise
        color = [0, 0, 0]  # Set noise to black
    else:
        color = colors[label]
    res[labels == label] = color  # Assign color to all pixels in the current cluster

# Reshape the clustered result back to the original image shape
res2 = res.reshape(img.shape)

# Display the result
cv.imshow('DBSCAN Clustered Image', res2)
cv.waitKey(0)
cv.destroyAllWindows()
