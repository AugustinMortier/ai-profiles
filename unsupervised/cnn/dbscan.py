import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import hdbscan
import matplotlib.pyplot as plt

# Path to the images
image_dir = 'images/rcs'
image_size = (256, 512)  # Resize images to a consistent size

# Step 1: Load and Preprocess the Dataset
images = []
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Load as grayscale
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    images.append(img_array)

images = np.array(images)
print(f"Loaded dataset shape: {images.shape}")

# 1. Load encoder
encoder_path = 'unsupervised/cnn/encoder.keras'
encoder = load_model(encoder_path)

# 2. Encode each image to get pixel-level features
encoded_images = encoder.predict(images)
print(f"Encoded images shape: {encoded_images.shape}")

# 3. Flatten spatial dimensions but keep feature channels intact
num_images, enc_height, enc_width, num_features = encoded_images.shape
encoded_images_flat = encoded_images.reshape((num_images * enc_height * enc_width, num_features))
print(f"Encoded pixel features shape for clustering: {encoded_images_flat.shape}")

# 4. Apply HDBSCAN to pixel features
# Adjust the `min_cluster_size` parameter as needed based on expected cluster sizes
clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
pixel_labels = clusterer.fit_predict(encoded_images_flat)
print("Clustering complete")

joblib.dump(clusterer, 'unsupervised/cnn/hdbscan.pkl')

# 5. Reshape pixel labels back into image form for visualization
pixel_labels_image_shape = pixel_labels.reshape((num_images, enc_height, enc_width))
print(f"Pixel-wise clustered image shape: {pixel_labels_image_shape.shape}")

# Optional: Plot a sample result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[0].reshape(image_size), cmap='gray')
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(pixel_labels_image_shape[0], cmap='tab20')
plt.title("HDBSCAN Clustered Image")
plt.show()
