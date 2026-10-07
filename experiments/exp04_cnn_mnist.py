# -*- coding: utf-8 -*-
"""
EXPERIMENT 04: Convolutional Neural Network (CNN) for Image Classification
Objective: Build, train, and evaluate a deep 2D CNN with feature extraction (Conv2D, MaxPool) and dense classification layers.
Dataset: MNIST Handwritten Digits
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Dataset Preparation
print("Loading and normalizing MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Build CNN Architecture
model = models.Sequential([
    # Feature Extractor Layer 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    # Feature Extractor Layer 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Classifier Head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compile, Train & Evaluate
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}% | Test Loss: {loss:.4f}")

# 4. Plotting training history
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', color='#6366f1')
plt.plot(history.history['val_loss'], label='Val Loss', color='#f43f5e')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc', color='#10b981')
plt.plot(history.history['val_accuracy'], label='Val Acc', color='#f59e0b')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
