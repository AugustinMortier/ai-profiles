import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from skimage import measure
from skimage.transform import resize

# input
image_path = 'images/rcs/AP_0-100-20000-0000001-A-2024-07-01.png'
method = 'kmeans' # 'kmeans', 'hdbscan'

# Paths to saved models
encoder_path = 'unsupervised/cnn/encoder.keras'
kmeans_path = 'unsupervised/cnn/kmeans.pkl'
hdbscan_path = 'unsupervised/cnn/hdbscan.pkl'

# Load the encoder and kmeans model
encoder = load_model(encoder_path)
if method == 'kmeans':
    cluster = joblib.load(kmeans_path)
elif method == 'hdbscan':
    cluster = joblib.load(hdbscan_path)

# Load and preprocess the image
image_size = (256, 512)  # Consistent with training size
img = load_img(image_path, target_size=image_size, color_mode='grayscale')
img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Encode the image to get feature representation
encoded_img = encoder.predict(img_array)[0]  # Remove batch dimension
print(f"Encoded image shape: {encoded_img.shape}")

# Step 1: Aggregate Encoded Features
print('encoded_img', np.shape(encoded_img))
aggregated_encoded_img = np.mean(encoded_img, axis=-1)  # Aggregated to single-channel (16, 32)
print('aggregated_encoded_img', np.shape(aggregated_encoded_img))

# Optional: Normalize for better visualization
aggregated_encoded_img = (aggregated_encoded_img - aggregated_encoded_img.min()) / (aggregated_encoded_img.max() - aggregated_encoded_img.min())

# Step 2: Flatten Encoded Features and Cluster
encoded_img_flat = encoded_img.reshape(-1, encoded_img.shape[-1])  # Flatten spatial dimensions for clustering
pixel_labels = cluster.predict(encoded_img_flat)  # Get cluster labels for each pixel

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

# Step 6: Plot the Results in a single figure with tighter layout
plt.figure(figsize=(16, 9))  # Adjust size as needed

# Set up a 2x2 grid with space for an additional 3x3 grid (for the 9 first features)
outer_grid = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1], wspace=0.1, hspace=0.1)

# Original image in top-left corner
ax1 = plt.subplot(outer_grid[0, 0])
ax1.imshow(img_array[0].reshape(image_size[0], image_size[1]), cmap='gray')
ax1.set_title("Original Image", fontsize=10)
ax1.axis('on')

# Overlay of clustered image in top-right corner
ax2 = plt.subplot(outer_grid[0, 1])
ax2.imshow(img_array[0].reshape(image_size[0], image_size[1]), cmap='gray')
ax2.imshow(clustered_image[:, :, :3], alpha=0.5)  # Overlay with transparency
ax2.set_title("Overlay of Clustered Image", fontsize=10)
ax2.axis('on')

# Upsampled Clustered Image in bottom-right corner
ax4 = plt.subplot(outer_grid[1, 1])
ax4.imshow(upsampled_pixel_labels, cmap='tab20')  # Adjust colors to distinguish clusters
ax4.set_title("Upsampled Clustered Image", fontsize=10)
ax4.axis('on')

# 3x3 grid of first 9 encoded features in the bottom-left corner
inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer_grid[1, 0], wspace=0.05, hspace=0.05)
for i in range(9):
    ax = plt.Subplot(plt.gcf(), inner_grid[i])
    ax.imshow(encoded_img[..., i], cmap='gray')
    ax.set_title(f'Feature {i+1}/{np.shape(encoded_img)[2]}', fontsize=8)
    ax.axis('off')
    plt.gcf().add_subplot(ax)

plt.tight_layout(pad=0.5)  # Further reduce padding around all panels
plt.show()