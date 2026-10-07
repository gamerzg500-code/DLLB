# -*- coding: utf-8 -*-
"""
EXPERIMENT 09: Denoising Autoencoder for Image Reconstruction
Objective: Train an autoencoder to corrupt inputs with Gaussian noise, encode latent representations, and reconstruct clean images.
Architectures: 
  - Primary: Deep Convolutional Denoising Autoencoder (Conv2D + MaxPool + UpSample)
  - Alternate: Fully Connected Dense Denoising Autoencoder (Dense bottleneck)
Dataset: MNIST Handwritten Digits
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load Data & Normalize
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 2. Add Additive Gaussian Noise & Clip to [0, 1]
noise_factor = 0.3
x_train_noisy = np.clip(x_train + np.random.normal(0, noise_factor, x_train.shape), 0.0, 1.0)
x_test_noisy = np.clip(x_test + np.random.normal(0, noise_factor, x_test.shape), 0.0, 1.0)

# 3. Build Convolutional Denoising Autoencoder
autoencoder = models.Sequential([
    # Encoder
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, padding='same'),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    # Decoder
    layers.UpSampling2D(2),
    layers.Conv2D(1, 3, activation='sigmoid', padding='same')
], name="Conv_Denoising_Autoencoder")

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

# 4. Train Autoencoder
print("Training Convolutional Denoising Autoencoder...")
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=5,
    batch_size=256,
    validation_data=(x_test_noisy, x_test)
)

# 5. Predict & Visualize 10 Samples
clean_preds = autoencoder.predict(x_test_noisy[:10])

plt.figure(figsize=(12, 4))
for i in range(10):
    # Row 1: Original Clean
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    if i == 0: plt.ylabel("Original", fontsize=10)
    plt.axis('off')

    # Row 2: Noisy Input
    plt.subplot(3, 10, 10 + i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    if i == 0: plt.ylabel("Noisy", fontsize=10)
    plt.axis('off')

    # Row 3: Denoised Output
    plt.subplot(3, 10, 20 + i + 1)
    plt.imshow(clean_preds[i].reshape(28, 28), cmap='gray')
    if i == 0: plt.ylabel("Reconstructed", fontsize=10)
    plt.axis('off')

plt.suptitle("Convolutional Denoising Autoencoder Reconstruction", fontsize=13)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
