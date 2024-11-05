import os
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

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

# Step 2: Define and Train an Autoencoder
# Encoder
input_img = Input(shape=(256, 512, 1))  # 256, 512, 1
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) # 256, 512, 32
x = MaxPooling2D((2, 2), padding='same')(x) # 128, 256, 32
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # 128, 256, 64
x = MaxPooling2D((2, 2), padding='same')(x) # 64, 128, 64

# Latent Space
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x) # 64, 128, 128
encoded = MaxPooling2D((2, 2), padding='same')(x) # 32, 64, 128

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) # 32, 64, 128
x = UpSampling2D((2, 2))(x) # 64, 128, 128
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) # 64, 128, 64
x = UpSampling2D((2, 2))(x) # 128, 256, 64
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) # 128, 256, 32
x = UpSampling2D((2, 2))(x) # 256, 512, 32
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # 256, 512, 1

# Model definition
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(images, images, epochs=10, batch_size=16, shuffle=True)

# Step 3: Extract Features and Cluster for Pixel-wise Clustering

# 1. Define the encoder model
encoder = Model(inputs=autoencoder.input, outputs=encoded)
encoder.save('unsupervised/cnn/encoder.keras')
