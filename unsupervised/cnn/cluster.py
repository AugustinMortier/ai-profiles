import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import measure
from skimage.transform import resize

# Paths to saved models and images
encoder_path = 'unsupervised/cnn/encoder.keras'
kmeans_path = 'unsupervised/cnn/kmeans.pkl'
image_path = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-02.png'

# Load the encoder and kmeans model
encoder = load_model(encoder_path)
kmeans = joblib.load(kmeans_path)

# Load and preprocess the image
image_size = (240, 624)  # Consistent with training size
img = load_img(image_path, target_size=image_size, color_mode='grayscale')
img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Encode the image to get feature representation
encoded_img = encoder.predict(img_array)[0]  # Remove batch dimension
print(f"Encoded image shape: {encoded_img.shape}")

# Step 1: Aggregate Encoded Features
aggregated_encoded_img = np.mean(encoded_img, axis=-1)  # Aggregated to single-channel (16, 32)

# Optional: Normalize for better visualization
aggregated_encoded_img = (aggregated_encoded_img - aggregated_encoded_img.min()) / (aggregated_encoded_img.max() - aggregated_encoded_img.min())

# Step 2: Flatten Encoded Features and Cluster
encoded_img_flat = encoded_img.reshape(-1, encoded_img.shape[-1])  # Flatten spatial dimensions for clustering
pixel_labels = kmeans.predict(encoded_img_flat)  # Get cluster labels for each pixel

# Reshape the cluster labels back to the spatial dimensions
pixel_labels_image_shape = pixel_labels.reshape(encoded_img.shape[0], encoded_img.shape[1])

# Step 3: Upsample the cluster labels to match the original image size
upsampled_pixel_labels = resize(pixel_labels_image_shape, (image_size[0], image_size[1]), order=0, preserve_range=True, anti_aliasing=False)
upsampled_pixel_labels = upsampled_pixel_labels.astype(int)  # Ensure the labels are integers

# Step 4: Create a colormap for cluster labels
unique_labels = np.unique(upsampled_pixel_labels)
colormap = plt.get_cmap('tab20', len(unique_labels))  # Get a colormap with enough colors

# Step 5: Create an overlay image with transparency
clustered_image = colormap(upsampled_pixel_labels)  # Apply the colormap
clustered_image[..., 3] = 0.5  # Set the alpha channel to 0.5 for transparency

# Step 6: Plot the Results
plt.figure(figsize=(12, 8))  # Adjust size for two rows

# Original image
plt.subplot(2, 2, 1)
plt.imshow(img_array[0].reshape(image_size[0], image_size[1]), cmap='gray')
plt.title("Original Image")

# Overlay Clustered Image on the Original Image
plt.subplot(2, 2, 2)
plt.imshow(img_array[0].reshape(image_size[0], image_size[1]), cmap='gray')
plt.imshow(clustered_image[:, :, :3], alpha=0.5)  # Overlay with transparency
plt.title("Overlay of Clustered Image")
plt.axis('off')  # Hide axes for better visual appeal

# Clustered Image
plt.subplot(2, 2, 4)
plt.imshow(upsampled_pixel_labels, cmap='tab20')  # Adjust colors to distinguish clusters
plt.title("Upsampled Clustered Image")

# Aggregated Encoded Features Image
plt.subplot(2, 2, 3)
plt.imshow(aggregated_encoded_img, cmap='gray')
plt.title("Aggregated Encoded Features")

plt.tight_layout()
plt.show()