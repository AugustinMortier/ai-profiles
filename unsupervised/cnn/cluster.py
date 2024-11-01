import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.decomposition import PCA
from skimage.transform import resize
import matplotlib.pyplot as plt

image_path = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png'

# Function to visualize clusters
def visualize_clusters(original_img, encoded_features, cluster_labels, num_clusters):
    # Create a color map for the clusters
    colors = plt.colormaps['hsv']  # Using HSV colormap
    clustered_image = np.zeros((*original_img.shape[0:2], 3))  # Empty image for clusters

    # Assign colors to clusters
    for cluster_id in range(num_clusters):
        clustered_image[cluster_labels == cluster_id] = colors(cluster_id / num_clusters)[:3]  # RGB values

    # Process encoded features with PCA
    encoded_flat = encoded_features.reshape(-1, encoded_features.shape[-1])  # Shape: (num_pixels, num_features)
    pca = PCA(n_components=3)
    encoded_pca = pca.fit_transform(encoded_flat)

    # Reshape PCA-reduced features to match the downsampled dimensions of the encoded features
    downsampled_height, downsampled_width = encoded_features.shape[0], encoded_features.shape[1]
    encoded_pca_img = encoded_pca.reshape(downsampled_height, downsampled_width, 3)
    
    # Resize PCA image to match original image dimensions
    encoded_pca_img = resize(encoded_pca_img, (original_img.shape[0], original_img.shape[1]), anti_aliasing=True)
    encoded_pca_img = (encoded_pca_img - encoded_pca_img.min()) / (encoded_pca_img.max() - encoded_pca_img.min())  # Normalize to 0-1

    # Plot original, encoded, and clustered images side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    # Encoded Features (PCA-reduced and resized)
    axs[1].imshow(encoded_pca_img)
    axs[1].set_title("Encoded Features (PCA)")
    axs[1].axis("off")

    # Clustered Image
    axs[2].imshow(clustered_image)
    axs[2].set_title("Clustered Image")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

image_size = (128, 256)  # Resize images to a consistent size

# Load and preprocess the new image
new_img = load_img(image_path, target_size=image_size, color_mode='rgb')
original_img_array = img_to_array(new_img) / 255.0  # Normalize to [0, 1]
new_img_array = np.expand_dims(original_img_array, axis=0)  # Add batch dimension

# Encode the new image using the encoder
encoder = tf.keras.models.load_model('unsupervised/cnn/encoder.keras')
encoded_features = encoder.predict(new_img_array)

# Remove the batch dimension from encoded features for PCA
encoded_features = encoded_features[0]

# Load KMeans model and predict cluster
kmeans = joblib.load('unsupervised/cnn/kmeans.pkl')
new_img_cluster = kmeans.predict(encoded_features.reshape(1, -1))

# Generate random cluster labels for visualization purposes (replace with actual cluster logic)
num_clusters = kmeans.n_clusters
cluster_labels = np.random.randint(0, num_clusters, original_img_array.shape[0] * original_img_array.shape[1])

# Reshape labels to match original image dimensions
cluster_labels = cluster_labels.reshape(original_img_array.shape[0], original_img_array.shape[1])

# Visualize the clusters
visualize_clusters(original_img_array, encoded_features, cluster_labels, num_clusters)
