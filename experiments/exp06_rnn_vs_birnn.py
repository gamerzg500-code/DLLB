# -*- coding: utf-8 -*-
"""
EXPERIMENT 06: Sequence Modeling: Simple RNN vs. Bidirectional RNN
Objective: Construct and compare unidirectional Recurrent Neural Network (SimpleRNN) with Bidirectional RNN for sequence prediction.
Dataset: Synthetic Sequential Arithmetic Progression Series
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1. Generate Synthetic Sequential Dataset
seq_len = 10
X, y = [], []
np.random.seed(42)
for _ in range(1000):
    s = np.random.randint(0, 100)
    seq = np.arange(s, s + seq_len)
    X.append(seq[:-1])
    y.append(seq[1:])
X = np.array(X, dtype=np.float32)[..., np.newaxis]
y = np.array(y, dtype=np.float32)[..., np.newaxis]

print(f"Dataset shape: X={X.shape}, y={y.shape}")

# 2. Unidirectional Simple RNN
print("\n--- Training Unidirectional Simple RNN ---")
rnn = Sequential([
    layers.SimpleRNN(32, return_sequences=True, input_shape=(seq_len - 1, 1)),
    layers.Dense(1)
])
rnn.compile(optimizer='adam', loss='mse')
h1 = rnn.fit(X, y, epochs=15, batch_size=32, verbose=0)

# 3. Bidirectional RNN
print("--- Training Bidirectional RNN ---")
birnn = Sequential([
    layers.Bidirectional(layers.SimpleRNN(32, return_sequences=True), input_shape=(seq_len - 1, 1)),
    layers.Dense(1)
])
birnn.compile(optimizer='adam', loss='mse')
h2 = birnn.fit(X, y, epochs=15, batch_size=32, verbose=0)

# 4. Comparative Evaluation
print("\n=== Performance Results ===")
print(f"Final MSE Loss (Standard Simple RNN): {h1.history['loss'][-1]:.6f}")
print(f"Final MSE Loss (Bidirectional RNN):   {h2.history['loss'][-1]:.6f}")

# Sample prediction test
sample = np.array([[[10], [11], [12], [13], [14], [15], [16], [17], [18]]], dtype=np.float32)
pred_rnn = rnn.predict(sample, verbose=0)[0].flatten()
pred_birnn = birnn.predict(sample, verbose=0)[0].flatten()
print(f"\nTarget Sequence:  {list(range(11, 20))}")
print(f"RNN Output:      {[round(float(v), 2) for v in pred_rnn]}")
print(f"BiRNN Output:    {[round(float(v), 2) for v in pred_birnn]}")
