# -*- coding: utf-8 -*-
"""
EXPERIMENT 02: Regularization Techniques & Data Augmentation on MNIST
Objective: Prevent overfitting using L1/L2 Weight Regularization, Dropout, Gaussian Noise, and ImageDataGenerator.
Dataset: MNIST Handwritten Digits
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Load & Preprocess MNIST Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)

# Add Gaussian Noise for robustness
noise = 0.05 * np.random.normal(size=x_train.shape)
x_train = np.clip(x_train + noise, 0.0, 1.0)

# 2. Real-time Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)

# 3. Model with L1/L2 Kernel Regularization and Dropout
model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# 4. Compile and Train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Training Regularized Model with Augmented Batches...")
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128, subset='training'),
    validation_data=datagen.flow(x_train, y_train, batch_size=128, subset='validation'),
    epochs=5
)
print("Training completed successfully!")
