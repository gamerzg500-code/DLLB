# -*- coding: utf-8 -*-
"""
EXPERIMENT 09 (Variant B): Fully-Connected Dense Denoising Autoencoder
Objective: Implement a lightweight feed-forward MLP bottleneck autoencoder for MNIST image denoising.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Dataset & Noise Injection
(x, _), (xt, _) = tf.keras.datasets.mnist.load_data()
x, xt = x / 255.0, xt / 255.0
xn = np.clip(x + 0.3 * np.random.randn(*x.shape), 0.0, 1.0)
xtn = np.clip(xt + 0.3 * np.random.randn(*xt.shape), 0.0, 1.0)

# 2. Dense Autoencoder Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='relu', name="bottleneck_encoder"),
    tf.keras.layers.Dense(784, activation='sigmoid', name="decoder_dense"),
    tf.keras.layers.Reshape((28, 28))
], name="Dense_Autoencoder")

# 3. Train
model.compile(optimizer='adam', loss='binary_crossentropy')
print("Training Dense Autoencoder...")
model.fit(xn, x, epochs=3, batch_size=256, validation_data=(xtn, xt))

# 4. Plot Comparison of 5 Samples
preds = model.predict(xtn[:5])
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(xtn[i], cmap='gray')
    plt.title(f"Noisy {i+1}")
    plt.axis('off')
    
    plt.subplot(2, 5, i + 6)
    plt.imshow(preds[i], cmap='gray')
    plt.title(f"Reconstructed {i+1}")
    plt.axis('off')

plt.suptitle("Dense Autoencoder Denoising Results", fontsize=12)
plt.tight_layout()
plt.show()
