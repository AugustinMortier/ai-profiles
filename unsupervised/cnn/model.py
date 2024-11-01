# Step 1: Load and Preprocess the Dataset

import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to the images
image_dir = 'images/rcs'
image_size = (128, 256)  # Resize images to a consistent size

# Load and preprocess images
images = []
for filename in os.listdir(image_dir):
    img_path = os.path.join(image_dir, filename)
    img = load_img(img_path, target_size=image_size, color_mode='rgb')
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    images.append(img_array)

images = np.array(images)
print(f"Loaded dataset shape: {images.shape}")



# Step 2: Define and Train an Autoencoder

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# Encoder
input_img = Input(shape=(128, 256, 3))
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
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Model definition
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(images, images, epochs=20, batch_size=16, shuffle=True)


# Step 3: Extract Features and Cluster

# Define encoder model
encoder = Model(inputs=autoencoder.input, outputs=encoded)
encoder.save('unsupervised/cnn/encoder.keras')

# Encode all images
encoded_images = encoder.predict(images)
encoded_images = encoded_images.reshape((encoded_images.shape[0], -1))  # Flatten to 2D

# Cluster the encoded features
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state=42, init='k-means++')
labels = kmeans.fit_predict(encoded_images)
joblib.dump(kmeans, 'unsupervised/cnn/kmeans.pkl')
#np.save('unsupervised/cnn/kmeans_labels.npy', labels)

# Visualize clustering results
import matplotlib.pyplot as plt

plt.scatter(encoded_images[:, 0], encoded_images[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.title("Clusters for Molecules, Aerosols, and Clouds")
plt.show()
