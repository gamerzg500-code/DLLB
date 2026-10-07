# -*- coding: utf-8 -*-
"""
EXPERIMENT 05: Long Short-Term Memory (LSTM) for Natural Language Sentiment Analysis
Objective: Process sequential text with dense word embedding representations and LSTM gating mechanisms.
Dataset: IMDB Large Movie Review Dataset (Binary Classification: Positive / Negative)
"""

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Dataset Parameters & Loading
vocab_size = 10000
maxlen = 200

print("Loading IMDB Movie Reviews (Top 10,000 words)...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences to uniform length
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

# 2. Build LSTM Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 3. Compile and Train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc * 100:.2f}% | Test Loss: {loss:.4f}")

# 4. Training Visualizations
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='#ef4444')
plt.plot(history.history['val_loss'], label='Validation Loss', color='#f59e0b')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#10b981')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#06b6d4')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
