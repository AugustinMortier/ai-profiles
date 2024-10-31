import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Function to visualize clusters
def visualize_clusters(original_img, encoded_img, cluster_labels, num_clusters):
    # Create a color map for the clusters
    colors = plt.cm.get_cmap('hsv', num_clusters)  # Using HSV colormap
    clustered_image = np.zeros((*original_img.shape[0:2], 3))  # Empty image for clusters

    # Assign colors to clusters
    for cluster_id in range(num_clusters):
        clustered_image[cluster_labels == cluster_id] = colors(cluster_id)[:3]  # RGB values

    # Plot original and clustered images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Clustered Image")
    plt.imshow(clustered_image)
    plt.axis('off')
    
    plt.show()

image_size = (128, 128)  # Resize images to a consistent size

# Load and preprocess the new image
new_image_path = 'unsupervised/images/rcs/AP_0-100-20000-0000001-A-2024-07-01.png'
new_img = load_img(new_image_path, target_size=image_size, color_mode='rgb')
original_img_array = img_to_array(new_img) / 255.0  # Normalize to [0, 1]
new_img_array = np.expand_dims(original_img_array, axis=0)  # Add batch dimension

# Encode the new image using the encoder
encoder = tf.keras.models.load_model('unsupervised/cnn/mymodel.keras')
encoded_new_img = encoder.predict(new_img_array)

# Flatten to match feature vector and convert to float32 if KMeans was trained on float32
encoded_new_img = encoded_new_img.flatten().astype(np.float32)  # Change to float32

# Reshape to 2D array (1, number_of_features)
encoded_new_img = np.reshape(encoded_new_img, (1, -1))

# Load KMeans model and predict cluster
kmeans = joblib.load('unsupervised/cnn/kmeans_model.pkl')
new_img_cluster = kmeans.predict(encoded_new_img)

# Get the labels for visualization (example: assuming the image is divided into a grid)
# Here, you need to get the cluster labels for each pixel or region in your original image
# For demonstration, I'll create random labels for the number of pixels in the image.
# Replace this with your actual clustering logic if you have it.
num_clusters = kmeans.n_clusters
cluster_labels = np.random.randint(0, num_clusters, original_img_array.shape[0] * original_img_array.shape[1])

# Reshape labels to match original image dimensions
cluster_labels = cluster_labels.reshape(original_img_array.shape[0], original_img_array.shape[1])

# Visualize the clusters
visualize_clusters(original_img_array, encoded_new_img, cluster_labels, num_clusters)
