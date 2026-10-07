# -*- coding: utf-8 -*-
"""
EXPERIMENT 10: Generative Adversarial Network (GAN) for Handwritten Digit Synthesis
Objective: Train a Generator and a Discriminator in a minimax adversarial game to generate realistic 28x28 digit images.
Dataset: MNIST Handwritten Digits
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 1. Dataset Preparation & Scaling to [-1, 1] for Tanh Activation
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape(-1, 28, 28, 1)

# 2. Generator Network (Latent vector z (dim 100) -> 28x28x1 image)
generator = tf.keras.Sequential([
    layers.Dense(7 * 7 * 64, input_dim=100),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"),
    layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh")
], name="Generator")

# 3. Discriminator Network (28x28x1 image -> Real/Fake binary probability)
discriminator = tf.keras.Sequential([
    layers.Conv2D(32, 5, strides=2, padding="same", input_shape=(28, 28, 1)),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")
], name="Discriminator")
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# 4. Combined GAN Model (Generator + Discriminator with frozen D weights)
discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator], name="GAN")
gan.compile(optimizer='adam', loss='binary_crossentropy')

# 5. Adversarial Training Loop
test_noise = np.random.normal(0, 1, (1, 100))
steps = 1501
batch_size = 128

print(f"Beginning GAN Adversarial Training for {steps} steps...")
for step in range(steps):
    # Train Discriminator with batch of real and fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fakes = generator.predict(noise, verbose=0)
    reals = x_train[np.random.randint(0, len(x_train), batch_size)]

    d_loss = discriminator.train_on_batch(
        np.concatenate([reals, fakes]),
        np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    )

    # Train Generator (via GAN) to fool Discriminator (target label = 1)
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if step % 500 == 0:
        print(f"Step {step:4d} | Discriminator Loss: {d_loss:.4f} | Generator Loss: {g_loss:.4f}")
        # Sample generated image
        img = generator.predict(test_noise, verbose=0)[0, :, :, 0]
        plt.figure(figsize=(3, 3))
        plt.imshow((img + 1) / 2, cmap='gray')
        plt.title(f"Generated at Step {step}")
        plt.axis('off')
        plt.show()

print("GAN Training procedure complete.")
