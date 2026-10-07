# -*- coding: utf-8 -*-
"""
EXPERIMENT 03: Comparison of Optimization Algorithms (SGD vs Momentum vs Adam)
Objective: Evaluate and compare learning dynamics, convergence rates, and loss curves of SGD, SGD with Momentum, and Adam.
Dataset: Non-linear XOR Problem
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
import matplotlib.pyplot as plt

# 1. XOR Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

def train_with_optimizer(opt):
    """Initializes and trains a model with a specified optimizer."""
    # Reset seed for fair comparison
    tf.random.set_seed(42)
    np.random.seed(42)
    model = Sequential([
        layers.Dense(4, input_dim=2, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy')
    history = model.fit(X, y, epochs=500, verbose=0)
    return history.history['loss']

# 2. Train with each optimizer
print("Benchmarking optimizers over 500 epochs...")
sgd_losses = train_with_optimizer(optimizers.SGD(learning_rate=0.1))
momentum_losses = train_with_optimizer(optimizers.SGD(learning_rate=0.1, momentum=0.9))
adam_losses = train_with_optimizer(optimizers.Adam(learning_rate=0.1))

print(f"Final Loss (SGD):      {sgd_losses[-1]:.6f}")
print(f"Final Loss (Momentum): {momentum_losses[-1]:.6f}")
print(f"Final Loss (Adam):     {adam_losses[-1]:.6f}")

# 3. Plotting the convergence trajectory
plt.figure(figsize=(10, 6))
plt.plot(sgd_losses, label='Standard SGD (lr=0.1)', color='#ef4444', linewidth=2)
plt.plot(momentum_losses, label='SGD + Momentum (lr=0.1, beta=0.9)', color='#f59e0b', linewidth=2)
plt.plot(adam_losses, label='Adam Optimizer (lr=0.1)', color='#10b981', linewidth=2)
plt.title('Loss Convergence Comparison Across Optimizers (XOR)', fontsize=14, pad=12)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Binary Crossentropy Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
