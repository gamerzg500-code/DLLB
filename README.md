# Deep Learning Laboratory: 10 Comprehensive Experiments Suite

An interactive, modern web portal and modular codebase containing all **10 Deep Learning Experiments** based on the lab curriculum and your notebook code.

---

## 🚀 How to Launch the Web Application

The interactive web page is located in this directory (`index.html`).

### Option 1: Direct Browser Launch
Simply double-click or open [index.html](file:///Users/ashish/Desktop/DL/index.html) in any modern browser (Chrome, Safari, Edge, Firefox).

### Option 2: Local HTTP Server (Already Running)
If you want to view it via local server:
```bash
python3 -m http.server 3456
```
Then visit: **[http://localhost:3456](http://localhost:3456)**

---

## 📂 Project Structure

```
DL/
├── index.html                           # Modern interactive web portal
├── style.css                            # Glassmorphic, dark/light theme stylesheet
├── app.js                               # 10 experiments data matrix & interactive features
├── Deep_Learning_All_10_Experiments.ipynb # Complete consolidated Jupyter notebook
├── README.md                            # Complete lab manual & execution guide
└── experiments/                         # Individual standalone Python scripts:
    ├── exp01_xor_mlp.py                 # Exp 1: Multi-Layer Perceptron (XOR)
    ├── exp02_regularization_mnist.py    # Exp 2: Regularization & Data Augmentation
    ├── exp03_optimizers_comparison.py   # Exp 3: Optimizers Comparison (SGD/Adam)
    ├── exp04_cnn_mnist.py               # Exp 4: CNN Image Classification (MNIST)
    ├── exp05_lstm_sentiment_imdb.py     # Exp 5: LSTM Sentiment Analysis (IMDB)
    ├── exp06_rnn_vs_birnn.py            # Exp 6: Simple RNN vs Bidirectional RNN
    ├── exp07_seq2seq_lstm.py            # Exp 7: Seq2Seq Encoder-Decoder (LSTM)
    ├── exp08_rbm_contrastive_divergence.py # Exp 8: RBM with Contrastive Divergence
    ├── exp09_denoising_autoencoder.py   # Exp 9: Convolutional Denoising Autoencoder
    ├── exp09b_dense_autoencoder.py      # Exp 9 (Alt): Dense Denoising Autoencoder
    └── exp10_dcgan_mnist.py             # Exp 10: Generative Adversarial Network (GAN)
```

---

## 🧪 Detailed Experiments Overview

| Exp # | Title | Model Architecture | Dataset | Key Concepts |
|:---:|:---|:---|:---|:---|
| **01** | **Multi-Layer Perceptron (MLP)** | Input(2) → Dense(4, ReLU) → Dense(1, Sigmoid) | XOR Truth Table | Non-linear boundary, Backpropagation, Binary Cross-Entropy |
| **02** | **Regularization & Augmentation** | Dense(256) + L1/L2 → Dropout(0.5) → Dense(128) + L2 → Dropout(0.3) | MNIST (28x28) | Weight penalties, Dropout, Gaussian noise, ImageDataGenerator |
| **03** | **Comparative Study of Optimizers** | Dense(4, Tanh) → Dense(1, Sigmoid) | Non-linear XOR | SGD vs. SGD with Momentum (β=0.9) vs. Adam |
| **04** | **Convolutional Neural Network (CNN)** | Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(64) | MNIST Digits | Spatial locality, Kernels, Feature maps, Pooling, Softmax |
| **05** | **LSTM for Sentiment Analysis** | Embedding(10000, 64) → LSTM(64) → Dropout(0.5) → Dense(1) | IMDB Reviews | Word Embeddings, Recurrent Gating, Long-term dependencies |
| **06** | **Simple RNN vs. Bidirectional RNN** | SimpleRNN(32) vs. Bidirectional(SimpleRNN(32)) | Synthetic Arithmetic Series | Forward vs Backward states, Sequential time-series forecasting |
| **07** | **Seq2Seq Encoder-Decoder with LSTM** | Encoder LSTM (state_h, state_c) ➔ Decoder LSTM (init_state) | Sequence Inversion / Reversal | Bottleneck representation, Recurrent state passing |
| **08** | **Restricted Boltzmann Machine (RBM)** | Visible (784) ⇄ Symmetric W ⇄ Hidden (128) | Binarized MNIST | Energy-based models, Gibbs sampling, Contrastive Divergence (CD-1) |
| **09** | **Denoising Autoencoders** | Conv2D(32) → MaxPool → Conv2D(32) ➔ UpSampling2D → Conv2D(1) | Noisy MNIST (σ=0.3) | Gaussian noise corruption, Unsupervised latent manifold projection |
| **10** | **Generative Adversarial Network (GAN)** | Generator (ConvTranspose, Tanh) vs. Discriminator (Conv, LeakyReLU) | MNIST Digits | Minimax zero-sum game, Adversarial training, Synthetic image synthesis |

---

## ⚡ Features of the Web Application

- **Search & Real-time Filter**: Type `/` to search by keywords, layers (`Conv2D`, `LSTM`, `Dropout`), or category.
- **Categorical Navigation**: Filter by *Foundational & MLP*, *Regularization & Vision*, *Sequential & NLP*, and *Unsupervised & Generative*.
- **Interactive Tabbed Views**:
  1. **Source Code**: Python syntax highlighting with line numbers, copy-to-clipboard, wrap toggles, and API chips.
  2. **Architecture & Graph**: Interactive tensor pipeline diagram and mathematical formulations.
  3. **Expected Output & Plots**: Terminal logs reproduction and interactive simulated plots/visualizations.
  4. **Theory & Viva Voce**: Comprehensive theory notes and common examination viva questions with expandable answers.
- **Progress Tracker**: Check off experiments as you complete or review them (saved in your browser).
- **Dark & Light Mode**: Built-in sleek cyber-dark theme and clean light theme.
- **Exporting Options**: One-click download of individual `.py` files or the complete Jupyter Notebook (`.ipynb`).
# DLLB
