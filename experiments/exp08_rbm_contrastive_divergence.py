# -*- coding: utf-8 -*-
"""
EXPERIMENT 08: Restricted Boltzmann Machine (RBM) with Contrastive Divergence
Objective: Train an energy-based unsupervised generative model with visible and hidden binary units using Gibbs sampling & CD-1.
Dataset: Binarized MNIST
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load & Binarize MNIST
print("Loading and binarizing MNIST dataset...")
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.reshape(-1, 784).astype('float32') / 255.0 > 0.5).astype('float32')
dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(64)

# 2. Initialize RBM Parameters
v_dim, h_dim = 784, 128
tf.random.set_seed(42)
W = tf.Variable(tf.random.normal([v_dim, h_dim], stddev=0.01), name="weights")
v_bias = tf.Variable(tf.zeros([v_dim]), name="visible_bias")
h_bias = tf.Variable(tf.zeros([h_dim]), name="hidden_bias")

def sample(p):
    """Bernoulli sampling based on conditional activation probabilities."""
    return tf.nn.relu(tf.sign(p - tf.random.uniform(tf.shape(p))))

# 3. Contrastive Divergence (CD-1) Training Loop
lr = 0.05
epochs = 3
epoch_losses = []

print(f"Beginning CD-1 training across {epochs} epochs...")
for epoch in range(epochs):
    total_loss = 0.0
    for v in dataset:
        # Positive Phase (Data driven)
        h_prob = tf.sigmoid(tf.matmul(v, W) + h_bias)
        h0 = sample(h_prob)

        # Negative Phase (Model reconstruction driven via 1-step Gibbs sampling)
        v1 = sample(tf.sigmoid(tf.matmul(h0, tf.transpose(W)) + v_bias))
        h1_prob = tf.sigmoid(tf.matmul(v1, W) + h_bias)

        # Gradient Updates
        bs = tf.cast(tf.shape(v)[0], tf.float32)
        w_grad = (tf.matmul(tf.transpose(v), h_prob) - tf.matmul(tf.transpose(v1), h1_prob)) / bs
        W.assign_add(lr * w_grad)
        v_bias.assign_add(lr * tf.reduce_mean(v - v1, axis=0))
        h_bias.assign_add(lr * tf.reduce_mean(h_prob - h1_prob, axis=0))

        total_loss += tf.reduce_mean(tf.square(v - v1)).numpy()

    epoch_avg_loss = total_loss / len(dataset)
    epoch_losses.append(epoch_avg_loss)
    print(f"Epoch {epoch + 1}/{epochs} | Reconstruction MSE Loss: {epoch_avg_loss:.4f}")

# 4. Plot Loss Progression
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='#8b5cf6', linewidth=2)
plt.title('RBM Training MSE Loss over Epochs (CD-1)', fontsize=13)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Reconstruction MSE Loss', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
