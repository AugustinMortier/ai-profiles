import os
import joblib
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Path to the images
image_dir = 'images/rcs'
image_size = (128, 256)  # Resize images to a consistent size

# Step 1: Load and Preprocess the Dataset
images = []
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    img = load_img(img_path, target_size=image_size, color_mode='grayscale')  # Load as grayscale
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    images.append(img_array)

images = np.array(images)
print(f"Loaded dataset shape: {images.shape}")

# Step 2: Define and Train an Autoencoder
# Encoder
input_img = Input(shape=(128, 256, 1))  # Input shape for grayscale images
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

# Latent Space
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Output shape for grayscale

# Model definition
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(images, images, epochs=10, batch_size=16, shuffle=True)

# Step 3: Extract Features and Cluster for Pixel-wise Clustering

# 1. Define the encoder model
encoder = Model(inputs=autoencoder.input, outputs=encoded)
encoder.save('unsupervised/cnn/encoder.keras')

# 2. Encode each image to get pixel-level features
encoded_images = encoder.predict(images)

# Check shape of encoded images
print(f"Encoded images shape: {encoded_images.shape}")  # Expected (283, 16, 32, 128) for example

# 3. Flatten spatial dimensions but keep feature channels intact
# This will give (num_images * height * width, num_features)
num_images, enc_height, enc_width, num_features = encoded_images.shape
encoded_images_flat = encoded_images.reshape((num_images * enc_height * enc_width, num_features))
print(f"Encoded pixel features shape for clustering: {encoded_images_flat.shape}")

# 4. Apply KMeans to pixel features
kmeans = KMeans(n_clusters=4)  # Assuming 6 clusters for molecules, aerosols, clouds, etc.
pixel_labels = kmeans.fit_predict(encoded_images_flat)  # Clusters all pixels independently
joblib.dump(kmeans, 'unsupervised/cnn/kmeans.pkl')

# 5. Reshape pixel labels back into image form for visualization
# After clustering, reshape pixel_labels to (num_images, enc_height, enc_width)
pixel_labels_image_shape = pixel_labels.reshape((num_images, enc_height, enc_width))
print(f"Pixel-wise clustered image shape: {pixel_labels_image_shape.shape}")
