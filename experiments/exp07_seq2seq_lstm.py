# -*- coding: utf-8 -*-
"""
EXPERIMENT 07: Sequence-to-Sequence (Seq2Seq) Encoder-Decoder with LSTM
Objective: Implement an Encoder-Decoder recurrent architecture using LSTM hidden states for sequence transformation (Sequence Reversal).
Model: Keras Functional API Encoder-Decoder LSTM
"""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# 1. Generate Synthetic Reversed Sequence Data
num_samples, timesteps, features = 1000, 5, 1
np.random.seed(42)
X = np.random.rand(num_samples, timesteps, features)
Y = np.flip(X, axis=1)  # Target is the input sequence in reverse order

# 2. Encoder Architecture
enc_in = Input(shape=(timesteps, features), name="encoder_input")
_, state_h, state_c = LSTM(32, return_state=True, name="encoder_lstm")(enc_in)
encoder_states = [state_h, state_c]

# 3. Decoder Architecture
dec_in = Input(shape=(timesteps, features), name="decoder_input")
dec_lstm = LSTM(32, return_sequences=True, name="decoder_lstm")(dec_in, initial_state=encoder_states)
dec_out = Dense(features, name="decoder_dense")(dec_lstm)

# 4. Assemble, Compile & Train Model
model = Model([enc_in, dec_in], dec_out, name="Seq2Seq_Autoencoder")
model.compile(optimizer='adam', loss='mse')
print("Seq2Seq Model Architecture:")
model.summary()

print("\nTraining Seq2Seq model on sequence reversal task...")
model.fit([X, Y], Y, epochs=10, batch_size=32, verbose=0)

# 5. Verification on Test Sequence
pred = model.predict([X[:1], Y[:1]], verbose=0)
print("\n--- Model Output Comparison ---")
print("Input Vector:    ", np.round(X[0].squeeze(), 4))
print("Predicted Output:", np.round(pred[0].squeeze(), 4))
print("Actual Vector:   ", np.round(Y[0].squeeze(), 4))
