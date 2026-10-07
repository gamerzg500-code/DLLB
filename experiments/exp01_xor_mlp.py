# -*- coding: utf-8 -*-
"""
EXPERIMENT 01: Multi-Layer Perceptron (MLP) for XOR Logic Gate
Objective: Construct, compile, and train a Multi-Layer Perceptron to solve the non-linearly separable XOR problem.
Framework: TensorFlow / Keras
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# 1. Prepare Dataset (XOR Truth Table)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# 2. Build Feedforward Neural Network (MLP)
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu', name="hidden_layer"),
    Dense(1, activation='sigmoid', name="output_layer")
])

# 3. Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train Model
print("Training MLP on XOR truth table...")
model.fit(X, y, epochs=1000, verbose=0)

# 5. Evaluate and Predict
loss, acc = model.evaluate(X, y, verbose=0)
print(f"\nFinal Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f}\n")

preds = (model.predict(X) > 0.5).astype(int)
print("--- Prediction Verification ---")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {preds[i][0]} (Actual: {y[i]})")
