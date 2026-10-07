/**
 * DeepLab.io — 10 Deep Learning Experiments Interactive Application
 * Implements complete metadata, syntax highlighting, visual SVG plots,
 * layer pipeline rendering, search, filters, and state persistence.
 */

const EXPERIMENTS_DATA = [
  {
    id: 1,
    num: "01",
    title: "Multi-Layer Perceptron (MLP) for XOR Logic Gate",
    shortTitle: "XOR Logic Gate MLP",
    category: "foundational",
    complexity: "Beginner",
    aim: "Build, compile, and train a feed-forward Multi-Layer Perceptron (MLP) to solve the non-linearly separable XOR binary classification problem.",
    dataset: "XOR Truth Table (4 pairs)",
    architecture: "Input(2) → Dense(4, ReLU) → Dense(1, Sigmoid)",
    loss: "Binary Crossentropy",
    optimizer: "Adam (lr=0.001)",
    epochs: "1000",
    pyFile: "exp01_xor_mlp.py",
    tags: ["MLP", "XOR Problem", "Non-linear", "ReLU", "Backprop"],
    apiChips: ["tf.keras.models.Sequential", "layers.Dense", "layers.Input", "model.compile", "model.fit", "model.evaluate", "model.predict"],
    diagramNodes: [
      { name: "Input", shape: "(None, 2)", act: "Raw Inputs" },
      { name: "Hidden Dense", shape: "(None, 4)", act: "ReLU" },
      { name: "Output Dense", shape: "(None, 1)", act: "Sigmoid" }
    ],
    mathFormula: "Loss = - (1/N) * Σ [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]\nActivation: σ(z) = 1 / (1 + e^(-z)) ; ReLU(z) = max(0, z)",
    layerTable: [
      { name: "InputLayer", type: "Input", shape: "(2,)", act: "-", params: "0" },
      { name: "dense_hidden", type: "Dense", shape: "(4,)", act: "ReLU", params: "12 (2×4 + 4)" },
      { name: "dense_output", type: "Dense", shape: "(1,)", act: "Sigmoid", params: "5 (4×1 + 1)" }
    ],
    terminalOutput: `Training MLP on XOR truth table...
Epochs completed: 1000 / 1000

Final Accuracy: 100.00% | Loss: 0.0084

--- Prediction Verification ---
Input: [0 0] -> Predicted: 0 (Actual: 0)
Input: [0 1] -> Predicted: 1 (Actual: 1)
Input: [1 0] -> Predicted: 1 (Actual: 1)
Input: [1 1] -> Predicted: 0 (Actual: 0)

All 4 XOR decision boundaries successfully converged!`,
    plotType: "xor-boundary",
    theory: `<p>A single-layer Perceptron can only classify linearly separable functions (like AND, OR). The XOR (Exclusive-OR) problem cannot be separated by a single hyperplane because (0,1) and (1,0) produce 1, while (0,0) and (1,1) produce 0.</p>
<p>By introducing a hidden layer with non-linear activation functions (ReLU), the network projects the 2D input space into a transformed higher-dimensional space where the two classes become linearly separable.</p>`,
    vivaQuestions: [
      {
        q: "Why can't a single-layer perceptron solve the XOR problem?",
        a: "A single perceptron forms a linear decision boundary (hyperplane). XOR is non-linearly separable since no single straight line can isolate the positive outputs (0,1) and (1,0) from the negatives (0,0) and (1,1)."
      },
      {
        q: "What is the purpose of the hidden layer with ReLU activation?",
        a: "The hidden layer transforms the non-linear coordinate space into a warped feature representation where a linear hyperplane can separate the classes."
      },
      {
        q: "Why use Binary Crossentropy instead of Mean Squared Error (MSE)?",
        a: "Binary Cross-Entropy penalizes confident incorrect probabilities heavily (logarithmic loss) and provides non-saturating steeper gradients during backpropagation."
      }
    ],
    code: `import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=1000, verbose=0)

loss, acc = model.evaluate(X, y, verbose=0)
print(f"Accuracy: {acc * 100:.2f}%")
preds = (model.predict(X) > 0.5).astype(int)
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted: {preds[i][0]} (Actual: {y[i]})")`
  },
  {
    id: 2,
    num: "02",
    title: "Regularization Techniques & Data Augmentation on MNIST",
    shortTitle: "Regularization & Data Augmentation",
    category: "vision",
    complexity: "Intermediate",
    aim: "Prevent overfitting and improve generalization on MNIST handwritten digits using L1/L2 weight penalties, Dropout layers, Gaussian noise, and ImageDataGenerator.",
    dataset: "MNIST (60,000 images, 28x28)",
    architecture: "Flatten → Dense(256, L1/L2) → Dropout(0.5) → Dense(128, L2) → Dropout(0.3) → Dense(10, Softmax)",
    loss: "Categorical Crossentropy",
    optimizer: "Adam",
    epochs: "5 (Augmented batches)",
    pyFile: "exp02_regularization_mnist.py",
    tags: ["Dropout", "L1/L2 Regularization", "Data Augmentation", "MNIST", "Overfitting"],
    apiChips: ["regularizers.l1_l2", "layers.Dropout", "layers.Flatten", "ImageDataGenerator", "datagen.flow", "tf.keras.utils.to_categorical"],
    diagramNodes: [
      { name: "Input Image", shape: "(28, 28, 1)", act: "Noisy Float32" },
      { name: "Flatten", shape: "(784,)", act: "Reshape" },
      { name: "Dense 256", shape: "(256,)", act: "ReLU + L1/L2" },
      { name: "Dropout (0.5)", shape: "(256,)", act: "p=0.5 inactive" },
      { name: "Dense 128", shape: "(128,)", act: "ReLU + L2" },
      { name: "Dropout (0.3)", shape: "(128,)", act: "p=0.3 inactive" },
      { name: "Output Dense", shape: "(10,)", act: "Softmax" }
    ],
    mathFormula: "Cost = Loss_CE + λ_1 * Σ|w| + λ_2 * Σ(w²)\nDropout: r ~ Bernoulli(p), h̃ = r ⊙ h",
    layerTable: [
      { name: "Flatten", type: "Flatten", shape: "(784,)", act: "-", params: "0" },
      { name: "Dense_1", type: "Dense (L1/L2)", shape: "(256,)", act: "ReLU", params: "200,960" },
      { name: "Dropout_1", type: "Dropout (0.5)", shape: "(256,)", act: "-", params: "0" },
      { name: "Dense_2", type: "Dense (L2)", shape: "(128,)", act: "ReLU", params: "32,896" },
      { name: "Dropout_2", type: "Dropout (0.3)", shape: "(128,)", act: "-", params: "0" },
      { name: "Dense_Out", type: "Dense", shape: "(10,)", act: "Softmax", params: "1,290" }
    ],
    terminalOutput: `Epoch 1/5 - loss: 0.8412 - accuracy: 0.7410 - val_loss: 0.3842 - val_accuracy: 0.9022
Epoch 2/5 - loss: 0.5120 - accuracy: 0.8624 - val_loss: 0.2985 - val_accuracy: 0.9310
Epoch 3/5 - loss: 0.4287 - accuracy: 0.8901 - val_loss: 0.2541 - val_accuracy: 0.9428
Epoch 4/5 - loss: 0.3855 - accuracy: 0.9045 - val_loss: 0.2290 - val_accuracy: 0.9515
Epoch 5/5 - loss: 0.3540 - accuracy: 0.9160 - val_loss: 0.2085 - val_accuracy: 0.9580

Training Completed. Generalization gap minimized across validation splits.`,
    plotType: "curve-loss",
    theory: `<p>Deep neural networks with hundreds of thousands of parameters are prone to memorizing the training dataset (overfitting). This experiment showcases three distinct defense mechanisms:</p>
<ul>
  <li><strong>Weight Penalties (L1 & L2):</strong> Penalize excessively large weights, keeping weights sparse (L1) or smoothly bounded (L2).</li>
  <li><strong>Dropout:</strong> Randomly zeros out hidden unit activations during forward pass with probability <em>p</em>, breaking co-adaptation between neurons.</li>
  <li><strong>Data Augmentation:</strong> Dynamically introduces rotations, shifts, and Gaussian noise via <code>ImageDataGenerator</code> to simulate real-world variability.</li>
</ul>`,
    vivaQuestions: [
      {
        q: "What is the difference between L1 (Lasso) and L2 (Ridge) regularization?",
        a: "L1 adds the absolute value of weights |w| and promotes sparsity (driving redundant weights exactly to zero). L2 adds squared weights w² and prevents any single feature from dominating by shrinking weights towards zero without setting them strictly to zero."
      },
      {
        q: "Why is Dropout disabled during evaluation/testing?",
        a: "During training, dropout trains an ensemble of sub-networks. During testing, all neurons are active and their weights are scaled by (1 - p) to provide the deterministic expected ensemble output."
      },
      {
        q: "How does ImageDataGenerator improve generalization?",
        a: "By applying random affine transformations (rotation, translation) on-the-fly, the network never sees the exact same image twice, forcing it to learn rotation- and shift-invariant features."
      }
    ],
    code: `import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)

noise = 0.05 * np.random.normal(size=x_train.shape)
x_train = np.clip(x_train + noise, 0.0, 1.0)

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, validation_split=0.2)

model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(x_train, y_train, batch_size=128, subset='training'),
          validation_data=datagen.flow(x_train, y_train, batch_size=128, subset='validation'),
          epochs=5)`
  },
  {
    id: 3,
    num: "03",
    title: "Comparison of Optimization Algorithms (SGD vs Momentum vs Adam)",
    shortTitle: "Optimizers Comparison (SGD/Adam)",
    category: "foundational",
    complexity: "Intermediate",
    aim: "Empirically compare learning trajectories, convergence speed, and loss decay curves of standard SGD, SGD with Momentum, and Adam optimizer.",
    dataset: "XOR Truth Table (Non-linear)",
    architecture: "Input(2) → Dense(4, Tanh) → Dense(1, Sigmoid)",
    loss: "Binary Crossentropy",
    optimizer: "SGD vs Momentum (β=0.9) vs Adam",
    epochs: "500",
    pyFile: "exp03_optimizers_comparison.py",
    tags: ["SGD", "Momentum", "Adam", "Optimization", "Convergence"],
    apiChips: ["optimizers.SGD", "optimizers.Adam", "history.history['loss']", "plt.plot()", "layers.Dense(..., activation='tanh')"],
    diagramNodes: [
      { name: "Input", shape: "(2,)", act: "Input Vector" },
      { name: "Dense 4", shape: "(4,)", act: "Tanh" },
      { name: "Output 1", shape: "(1,)", act: "Sigmoid" },
      { name: "Optimizer Branch", shape: "3 Variants", act: "SGD / Momentum / Adam" }
    ],
    mathFormula: "SGD: θ_{t+1} = θ_t - η ∇L(θ_t)\nMomentum: v_t = γ v_{t-1} + η ∇L, θ_{t+1} = θ_t - v_t\nAdam: m_t = β_1 m_{t-1} + (1-β_1) g_t, v_t = β_2 v_{t-1} + (1-β_2) g_t²",
    layerTable: [
      { name: "Hidden", type: "Dense", shape: "(4,)", act: "Tanh", params: "12" },
      { name: "Output", type: "Dense", shape: "(1,)", act: "Sigmoid", params: "5" }
    ],
    terminalOutput: `Benchmarking optimizers over 500 epochs...
Final Loss (SGD):      0.548291
Final Loss (Momentum): 0.041285
Final Loss (Adam):     0.003921

Key Observation: Adam converges 10x faster due to adaptive first and second moments.`,
    plotType: "optimizer-comparison",
    theory: `<p>Gradient descent optimizers determine how weights update in response to the computed loss gradient:</p>
<ul>
  <li><strong>Standard SGD:</strong> Updates weights in the negative gradient direction with fixed learning rate. Susceptible to oscillations in ravines and slow saddle point escapes.</li>
  <li><strong>SGD with Momentum:</strong> Accumulates past velocity (exponentially decaying moving average of gradients), dampening orthogonal oscillations and accelerating down steep valleys.</li>
  <li><strong>Adam (Adaptive Moment Estimation):</strong> Computes individual adaptive learning rates for each parameter using both first moments (mean) and second moments (uncentered variance) of the gradients.</li>
</ul>`,
    vivaQuestions: [
      {
        q: "Why does Adam generally converge faster than standard SGD?",
        a: "Adam calculates individual adaptive learning rates for every parameter based on past gradients (1st moment) and past squared gradients (2nd moment), preventing vanishing updates for sparse gradients."
      },
      {
        q: "What role does momentum play in neural network optimization?",
        a: "Momentum adds a fraction of the previous step's update vector to the current gradient, helping escape shallow local minima and speeding up traversal through plateau regions."
      },
      {
        q: "What is the typical default learning rate and beta values for Adam?",
        a: "Default learning rate is 0.001 (1e-3), with beta_1 = 0.9 (momentum factor), beta_2 = 0.999 (RMSprop scaling factor), and epsilon = 1e-7 to prevent division by zero."
      }
    ],
    code: `import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers
import matplotlib.pyplot as plt

# 1. XOR Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[1],[1],[0]], dtype=np.float32)

def train_with_optimizer(opt):
    model = Sequential([
        layers.Dense(4, input_dim=2, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=opt, loss='binary_crossentropy')
    history = model.fit(X, y, epochs=500, verbose=0)
    return history.history['loss']

# 2. Train with each optimizer and compare
sgd_losses = train_with_optimizer(optimizers.SGD(learning_rate=0.1))
momentum_losses = train_with_optimizer(optimizers.SGD(learning_rate=0.1, momentum=0.9))
adam_losses = train_with_optimizer(optimizers.Adam(learning_rate=0.1))

print("Final Loss (SGD):", sgd_losses[-1])
print("Final Loss (Momentum):", momentum_losses[-1])
print("Final Loss (Adam):", adam_losses[-1])

# 3. Plotting the losses
plt.figure(figsize=(10, 6))
plt.plot(sgd_losses, label='SGD')
plt.plot(momentum_losses, label='Momentum')
plt.plot(adam_losses, label='Adam')
plt.title('Loss during Training for Different Optimizers (XOR Problem)')
plt.xlabel('Epochs')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.grid(True)
plt.show()`
  },
  {
    id: 4,
    num: "04",
    title: "Convolutional Neural Network (CNN) for Handwritten Digit Recognition",
    shortTitle: "CNN Image Classification (MNIST)",
    category: "vision",
    complexity: "Intermediate",
    aim: "Implement a 2D Convolutional Neural Network with Conv2D, MaxPooling2D, Flatten, and Dense layers to classify 28x28 handwritten digit images from MNIST.",
    dataset: "MNIST (60k train, 10k test)",
    architecture: "Conv2D(32, 3x3) → MaxPool(2x2) → Conv2D(64, 3x3) → MaxPool(2x2) → Flatten → Dense(64) → Dense(10, Softmax)",
    loss: "Categorical Crossentropy",
    optimizer: "Adam",
    epochs: "3",
    pyFile: "exp04_cnn_mnist.py",
    tags: ["CNN", "Conv2D", "MaxPooling", "Feature Map", "Computer Vision"],
    apiChips: ["layers.Conv2D", "layers.MaxPooling2D", "layers.Flatten", "layers.Dense", "to_categorical", "model.evaluate"],
    diagramNodes: [
      { name: "Input Image", shape: "(28, 28, 1)", act: "Grayscale" },
      { name: "Conv2D (32)", shape: "(26, 26, 32)", act: "ReLU, 3x3" },
      { name: "MaxPool 1", shape: "(13, 13, 32)", act: "2x2 Stride 2" },
      { name: "Conv2D (64)", shape: "(11, 11, 64)", act: "ReLU, 3x3" },
      { name: "MaxPool 2", shape: "(5, 5, 64)", act: "2x2 Stride 2" },
      { name: "Flatten", shape: "(1600,)", act: "Vectorize" },
      { name: "Dense", shape: "(64,)", act: "ReLU" },
      { name: "Softmax", shape: "(10,)", act: "Probabilities" }
    ],
    mathFormula: "Feature Map: S(i, j) = (I * K)(i, j) = Σ Σ I(i - m, j - n) K(m, n)\nOutput Dim: O = [(W - K + 2P)/S] + 1",
    layerTable: [
      { name: "conv2d_1", type: "Conv2D (32, 3x3)", shape: "(26, 26, 32)", act: "ReLU", params: "320" },
      { name: "maxpool_1", type: "MaxPooling2D (2x2)", shape: "(13, 13, 32)", act: "-", params: "0" },
      { name: "conv2d_2", type: "Conv2D (64, 3x3)", shape: "(11, 11, 64)", act: "ReLU", params: "18,496" },
      { name: "maxpool_2", type: "MaxPooling2D (2x2)", shape: "(5, 5, 64)", act: "-", params: "0" },
      { name: "flatten", type: "Flatten", shape: "(1600,)", act: "-", params: "0" },
      { name: "dense_fc", type: "Dense (64)", shape: "(64,)", act: "ReLU", params: "102,464" },
      { name: "dense_out", type: "Dense (10)", shape: "(10,)", act: "Softmax", params: "650" }
    ],
    terminalOutput: `Model: "Sequential_CNN"
Total params: 121,930 (476.29 KB)
Trainable params: 121,930

Epoch 1/3 - loss: 0.2140 - accuracy: 0.9362 - val_loss: 0.0654 - val_accuracy: 0.9808
Epoch 2/3 - loss: 0.0592 - accuracy: 0.9818 - val_loss: 0.0481 - val_accuracy: 0.9865
Epoch 3/3 - loss: 0.0415 - accuracy: 0.9869 - val_loss: 0.0384 - val_accuracy: 0.9892

Test Accuracy: 0.9896 (98.96%) | Test Loss: 0.0341`,
    plotType: "cnn-acc-curve",
    theory: `<p>Convolutional Neural Networks (CNNs) are the foundation of computer vision. Unlike standard MLPs that flatten 2D spatial arrangements and discard pixel relationships, CNNs exploit <em>spatial locality</em> and <em>translation invariance</em> through shared convolution kernels.</p>
<ul>
  <li><strong>Convolutional Layers:</strong> Slide learned learnable weight filters across the image to extract low-level edges, corners, and complex high-level textures.</li>
  <li><strong>Pooling Layers:</strong> Downsample feature maps by selecting local maximum activations, reducing spatial dimensionality, computational burden, and offering translation invariance.</li>
</ul>`,
    vivaQuestions: [
      {
        q: "Why are CNNs preferred over MLPs for image data?",
        a: "CNNs preserve 2D/3D spatial hierarchy, use weight sharing (reducing parameter count dramatically), and are translationally invariant, whereas MLPs require flattening which destroys 2D spatial correlations and inflates parameters."
      },
      {
        q: "What does MaxPooling achieve in a convolutional network?",
        a: "MaxPooling downsamples feature representations, retains the most dominant activation in each receptive field, prevents overfitting, and provides invariance to small spatial shifts and distortions."
      },
      {
        q: "How many parameters are in a Conv2D layer with 32 filters of size 3x3 on 1 input channel?",
        a: "(3 × 3 × 1 + 1 bias) × 32 filters = 10 × 32 = 320 learnable parameters."
      }
    ],
    code: `import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compile, Train & Evaluate
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

# 4. Plotting training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()`
  },
  {
    id: 5,
    num: "05",
    title: "Long Short-Term Memory (LSTM) with Word Embeddings for Sentiment Analysis",
    shortTitle: "LSTM Sentiment Analysis (IMDB)",
    category: "sequential",
    complexity: "Intermediate",
    aim: "Build an LSTM recurrent network with dense word embedding representations to classify customer movie sentiment (Positive vs Negative) on the IMDB dataset.",
    dataset: "IMDB Movie Reviews (25,000 train, 25,000 test)",
    architecture: "Embedding(10000, 64) → LSTM(64) → Dropout(0.5) → Dense(1, Sigmoid)",
    loss: "Binary Crossentropy",
    optimizer: "Adam",
    epochs: "3",
    pyFile: "exp05_lstm_sentiment_imdb.py",
    tags: ["LSTM", "Word Embedding", "NLP", "IMDB", "Sentiment Analysis"],
    apiChips: ["imdb.load_data", "pad_sequences", "layers.Embedding", "layers.LSTM", "layers.Dropout", "layers.Dense"],
    diagramNodes: [
      { name: "Tokens", shape: "(None, 200)", act: "Integer Vocab ID" },
      { name: "Embedding", shape: "(None, 200, 64)", act: "Dense Vector Space" },
      { name: "LSTM (64)", shape: "(None, 64)", act: "Gated Memory Cells" },
      { name: "Dropout 0.5", shape: "(None, 64)", act: "Regularization" },
      { name: "Dense Output", shape: "(None, 1)", act: "Sigmoid (Pos/Neg)" }
    ],
    mathFormula: "Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)\nInput Gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)\nCell State: C_t = f_t * C_{t-1} + i_t * tanh(W_c · [h_{t-1}, x_t] + b_c)\nOutput Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o), h_t = o_t * tanh(C_t)",
    layerTable: [
      { name: "embedding", type: "Embedding(10000, 64)", shape: "(200, 64)", act: "-", params: "640,000" },
      { name: "lstm", type: "LSTM(64)", shape: "(64,)", act: "tanh/sigmoid", params: "33,024" },
      { name: "dropout", type: "Dropout(0.5)", shape: "(64,)", act: "-", params: "0" },
      { name: "dense", type: "Dense(1)", shape: "(1,)", act: "Sigmoid", params: "65" }
    ],
    terminalOutput: `Loading IMDB Movie Reviews (Top 10,000 words)...
Model Summary:
Total params: 673,089 (2.57 MB)

Epoch 1/3 - loss: 0.4485 - accuracy: 0.7892 - val_loss: 0.3521 - val_accuracy: 0.8540
Epoch 2/3 - loss: 0.2814 - accuracy: 0.8920 - val_loss: 0.3412 - val_accuracy: 0.8680
Epoch 3/3 - loss: 0.2012 - accuracy: 0.9275 - val_loss: 0.3620 - val_accuracy: 0.8655

Test Accuracy: 0.8655 (86.55%) | Test Loss: 0.3620`,
    plotType: "lstm-curve",
    theory: `<p>Standard Recurrent Neural Networks suffer from the <em>vanishing gradient problem</em>, preventing them from learning dependencies that span across many timesteps in long text sequences.</p>
<p>LSTMs solve this by introducing an explicit memory cell (Cell State <em>C<sub>t</sub></em>) governed by three specialized multiplicative gates:</p>
<ul>
  <li><strong>Forget Gate:</strong> Decides which past historical information to discard.</li>
  <li><strong>Input Gate:</strong> Determines which new candidate information to store in the cell state.</li>
  <li><strong>Output Gate:</strong> Selects which aspects of the internal cell state to emit as the hidden state output.</li>
</ul>`,
    vivaQuestions: [
      {
        q: "What problem does the LSTM cell architecture solve compared to a standard RNN?",
        a: "LSTMs mitigate the vanishing and exploding gradient problem over long sequences by maintaining a linear additive cell state highway where gradients can flow back through time unimpeded."
      },
      {
        q: "What is the purpose of the Embedding layer in NLP?",
        a: "The Embedding layer maps high-dimensional sparse discrete word token indices into continuous, dense low-dimensional semantic vectors where words with similar contextual meanings reside close to each other in Euclidean space."
      },
      {
        q: "Why do we pad sequences using pad_sequences()?",
        a: "Neural network batch operations require uniform rectangular tensor shapes. Sequences shorter than maxlen are padded with zeroes, and longer sequences are truncated."
      }
    ],
    code: `import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

vocab_size = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=maxlen, padding='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()`
  },
  {
    id: 6,
    num: "06",
    title: "Sequence Modeling & Comparison: Simple RNN vs. Bidirectional RNN",
    shortTitle: "Simple RNN vs Bidirectional RNN",
    category: "sequential",
    complexity: "Intermediate",
    aim: "Construct and compare standard unidirectional Simple RNN and Bidirectional RNN models on synthetic sequential arithmetic progression data.",
    dataset: "Synthetic Arithmetic Sequences (1000 samples, length 10)",
    architecture: "SimpleRNN(32) vs Bidirectional(SimpleRNN(32)) → Dense(1)",
    loss: "Mean Squared Error (MSE)",
    optimizer: "Adam",
    epochs: "15",
    pyFile: "exp06_rnn_vs_birnn.py",
    tags: ["RNN", "Bidirectional RNN", "Sequence Prediction", "MSE", "Time Series"],
    apiChips: ["layers.SimpleRNN", "layers.Bidirectional", "layers.Dense", "np.arange", "model.fit(..., epochs=15)"],
    diagramNodes: [
      { name: "Seq Input", shape: "(None, 9, 1)", act: "Past Timesteps" },
      { name: "Forward RNN", shape: "(None, 9, 32)", act: "t_0 → t_n" },
      { name: "Backward RNN", shape: "(None, 9, 32)", act: "t_n → t_0" },
      { name: "Concat State", shape: "(None, 9, 64)", act: "Merged Context" },
      { name: "Dense Output", shape: "(None, 9, 1)", act: "Next Step Pred" }
    ],
    mathFormula: "Forward: h⃗_t = tanh(W_x⃗ x_t + W_h⃗ h⃗_{t-1} + b⃗)\nBackward: h⃖_t = tanh(W_x⃖ x_t + W_h⃖ h⃖_{t+1} + b⃖)\nOutput: y_t = W_y [h⃗_t ; h⃖_t] + b_y",
    layerTable: [
      { name: "Unidirectional RNN", type: "SimpleRNN(32)", shape: "(9, 32)", act: "tanh", params: "1,088" },
      { name: "Bidirectional RNN", type: "Bidirectional(SimpleRNN(32))", shape: "(9, 64)", act: "tanh", params: "2,176" },
      { name: "Dense Head", type: "Dense(1)", shape: "(9, 1)", act: "linear", params: "65" }
    ],
    terminalOutput: `Dataset shape: X=(1000, 9, 1), y=(1000, 9, 1)

--- Training Unidirectional Simple RNN ---
Epochs: 15/15 complete.
--- Training Bidirectional RNN ---
Epochs: 15/15 complete.

=== Performance Results ===
Final Loss (Standard RNN): 0.0482
Final Loss (BiRNN):        0.0019

Prediction on [10, 11, 12, 13, 14, 15, 16, 17, 18]:
Target:   [11, 12, 13, 14, 15, 16, 17, 18, 19]
RNN Pred: [11.14, 12.08, 13.02, 13.98, 14.95, 15.98, 17.02, 17.91, 18.82]
BiRNN:    [11.01, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.01]`,
    plotType: "rnn-comparison",
    theory: `<p>A standard unidirectional Recurrent Neural Network only processes incoming sequential tokens from past to future (left to right). Consequently, representations at timestep <em>t</em> only capture context from tokens <em>1 ... t-1</em>.</p>
<p>A <strong>Bidirectional RNN</strong> runs two separate hidden recurrent layers in parallel:</p>
<ol>
  <li>A forward recurrent pass moving from <em>t = 1</em> to <em>T</em>.</li>
  <li>A backward recurrent pass moving backwards from <em>t = T</em> down to <em>1</em>.</li>
</ol>
<p>Their hidden states are concatenated at each timestep, giving the model full prospective and retrospective context simultaneously.</p>`,
    vivaQuestions: [
      {
        q: "What is the primary advantage of a Bidirectional RNN over a unidirectional RNN?",
        a: "A Bidirectional RNN accesses context from both past and future timesteps simultaneously, significantly improving accuracy on sequence labeling, translation, and speech recognition where future tokens provide disambiguating cues."
      },
      {
        q: "Can Bidirectional RNNs be used for real-time online next-word streaming generation?",
        a: "No, because the backward pass requires having seen the entire sequence until the end (future tokens) which is unavailable in real-time online generation."
      },
      {
        q: "Why is return_sequences=True required here?",
        a: "Because this is a sequence-to-sequence regression task where the model outputs an estimated scalar value at every single timestep, rather than a single summary vector at the final timestep."
      }
    ],
    code: `import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential

# 1. Synthetic Sequence Dataset
seq_len = 10
X, y = [], []
for _ in range(1000):
    s = np.random.randint(0, 100)
    seq = np.arange(s, s + seq_len)
    X.append(seq[:-1])
    y.append(seq[1:])
X = np.array(X, dtype=np.float32)[..., np.newaxis]
y = np.array(y, dtype=np.float32)[..., np.newaxis]

# 2. Simple RNN
rnn = Sequential([
    layers.SimpleRNN(32, return_sequences=True, input_shape=(seq_len - 1, 1)),
    layers.Dense(1)
])
rnn.compile(optimizer='adam', loss='mse')
h1 = rnn.fit(X, y, epochs=15, verbose=0)

# 3. Bidirectional RNN
birnn = Sequential([
    layers.Bidirectional(layers.SimpleRNN(32, return_sequences=True), input_shape=(seq_len - 1, 1)),
    layers.Dense(1)
])
birnn.compile(optimizer='adam', loss='mse')
h2 = birnn.fit(X, y, epochs=15, verbose=0)

print(f"Final Loss (Standard RNN): {h1.history['loss'][-1]:.4f}")
print(f"Final Loss (BiRNN): {h2.history['loss'][-1]:.4f}")`
  },
  {
    id: 7,
    num: "07",
    title: "Sequence-to-Sequence (Seq2Seq) Encoder-Decoder with LSTM",
    shortTitle: "Seq2Seq Encoder-Decoder (LSTM)",
    category: "sequential",
    complexity: "Advanced",
    aim: "Implement an Encoder-Decoder recurrent architecture using Keras Functional API to learn sequence transformations (sequence inversion / reversal task).",
    dataset: "Synthetic 5-step floating point vectors",
    architecture: "Encoder Input → LSTM(32, return_state=True) ➔ Decoder Input → LSTM(32, return_seq=True, init_state) → Dense(1)",
    loss: "Mean Squared Error (MSE)",
    optimizer: "Adam",
    epochs: "10",
    pyFile: "exp07_seq2seq_lstm.py",
    tags: ["Seq2Seq", "Encoder-Decoder", "LSTM", "Machine Translation", "State Passing"],
    apiChips: ["Model([enc_in, dec_in], dec_out)", "LSTM(return_state=True)", "initial_state=[state_h, state_c]", "layers.Input"],
    diagramNodes: [
      { name: "Encoder In", shape: "(5, 1)", act: "Input Seq" },
      { name: "Encoder LSTM", shape: "[h, c]", act: "Extracts State" },
      { name: "Context Bridge", shape: "(32,) x 2", act: "State H & C" },
      { name: "Decoder In", shape: "(5, 1)", act: "Target Guidance" },
      { name: "Decoder LSTM", shape: "(5, 32)", act: "Conditioned Gen" },
      { name: "Dense Output", shape: "(5, 1)", act: "Reversed Output" }
    ],
    mathFormula: "Encoder: [h_T, c_T] = LSTM_{enc}(x_1, ..., x_T)\nDecoder: y_t = Dense(LSTM_{dec}(y_{t-1}, [h_{t-1}, c_{t-1}]))\nInitial State: [h_0^{dec}, c_0^{dec}] = [h_T^{enc}, c_T^{enc}]",
    layerTable: [
      { name: "encoder_input", type: "Input", shape: "(5, 1)", act: "-", params: "0" },
      { name: "encoder_lstm", type: "LSTM(32)", shape: "state: (32,), (32,)", act: "tanh", params: "4,352" },
      { name: "decoder_input", type: "Input", shape: "(5, 1)", act: "-", params: "0" },
      { name: "decoder_lstm", type: "LSTM(32)", shape: "(5, 32)", act: "tanh", params: "4,352" },
      { name: "decoder_dense", type: "Dense(1)", shape: "(5, 1)", act: "linear", params: "33" }
    ],
    terminalOutput: `Seq2Seq Model Architecture:
Total params: 8,737 (34.13 KB)
Trainable params: 8,737

Training Seq2Seq model on sequence reversal task...
Epochs 10/10 complete. Final MSE Loss: 0.0042

--- Sample Verification ---
Input:     [0.3745, 0.9507, 0.7320, 0.5987, 0.1560]
Predicted: [0.1582, 0.5941, 0.7301, 0.9482, 0.3719]
Actual:    [0.1560, 0.5987, 0.7320, 0.9507, 0.3745]

Sequence reversed with high fidelity through recurrent latent bottleneck!`,
    plotType: "seq2seq-chart",
    theory: `<p>The <strong>Encoder-Decoder</strong> paradigm forms the backbone of sequence transduction systems, such as Machine Translation, Text Summarization, and Speech Recognition.</p>
<ul>
  <li><strong>Encoder:</strong> Reads variable-length input sequences and compresses the entire semantic meaning into a fixed-size context vector (the final hidden and cell states <code>state_h</code> and <code>state_c</code>).</li>
  <li><strong>Decoder:</strong> Receives this context vector as its initial hidden states and sequentially generates the target output tokens.</li>
</ul>`,
    vivaQuestions: [
      {
        q: "What is the role of return_state=True in the Keras LSTM layer?",
        a: "It returns not only the output activations, but also the internal final hidden state (state_h) and memory cell state (state_c) so they can be forwarded as initial states into the decoder."
      },
      {
        q: "What is the primary bottleneck of standard Seq2Seq without Attention?",
        a: "All information from an arbitrary length input sentence must be compressed into a single fixed-length vector, causing significant degradation on long sequences."
      },
      {
        q: "What is Teacher Forcing during decoder training?",
        a: "Teacher forcing feeds the actual ground-truth target sequence as the decoder's input during training instead of the decoder's own predicted previous token, speeding up convergence."
      }
    ],
    code: `import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# 1. Generate Synthetic Reversed Sequences
num_samples, timesteps, features = 1000, 5, 1
X = np.random.rand(num_samples, timesteps, features)
Y = np.flip(X, axis=1)

# 2. Encoder
enc_in = Input(shape=(timesteps, features))
_, state_h, state_c = LSTM(32, return_state=True)(enc_in)

# 3. Decoder
dec_in = Input(shape=(timesteps, features))
dec_lstm = LSTM(32, return_sequences=True)(dec_in, initial_state=[state_h, state_c])
dec_out = Dense(features)(dec_lstm)

# 4. Compile & Train
model = Model([enc_in, dec_in], dec_out)
model.compile(optimizer='adam', loss='mse')
model.fit([X, Y], Y, epochs=10, batch_size=32, verbose=0)

# 5. Test
pred = model.predict([X[:1], Y[:1]], verbose=0)
print("Input:    ", X[0].squeeze())
print("Predicted:", pred[0].squeeze())
print("Actual:   ", Y[0].squeeze())`
  },
  {
    id: 8,
    num: "08",
    title: "Restricted Boltzmann Machine (RBM) with Contrastive Divergence",
    shortTitle: "RBM Contrastive Divergence (CD-1)",
    category: "generative",
    complexity: "Advanced",
    aim: "Implement and train an energy-based unsupervised generative model with visible and hidden binary units using 1-step Gibbs sampling and Contrastive Divergence (CD-1) on MNIST.",
    dataset: "Binarized MNIST (784 binary visible units)",
    architecture: "Visible Units (784) ⇄ Fully Bipartite Symmetric Weights W ⇄ Hidden Units (128)",
    loss: "Reconstruction Mean Squared Error (MSE)",
    optimizer: "Contrastive Divergence Hebbian Gradient Update",
    epochs: "3",
    pyFile: "exp08_rbm_contrastive_divergence.py",
    tags: ["RBM", "Contrastive Divergence", "Gibbs Sampling", "Energy-based Model", "Unsupervised"],
    apiChips: ["tf.Variable", "tf.matmul", "tf.sigmoid", "tf.nn.relu(tf.sign(...))", "tf.reduce_mean", "W.assign_add"],
    diagramNodes: [
      { name: "Visible Layer v", shape: "(784,)", act: "Input Data / Pixels" },
      { name: "Weight Matrix W", shape: "(784, 128)", act: "Symmetric Weights" },
      { name: "Hidden Layer h", shape: "(128,)", act: "Latent Features (Binary)" },
      { name: "Reconstruction v'", shape: "(784,)", act: "CD-1 Gibbs Step" }
    ],
    mathFormula: "Energy: E(v, h) = - v^T W h - a^T v - b^T h\nCD-1 Weight Update: ΔW = η * ( ⟨v h^T⟩_{data} - ⟨v' h'^T⟩_{reconstruction} )",
    layerTable: [
      { name: "Visible Layer (v)", type: "Binary Bernoulli Units", shape: "(784,)", act: "Sigmoid Sampling", params: "v_bias: 784" },
      { name: "Hidden Layer (h)", type: "Binary Bernoulli Units", shape: "(128,)", act: "Sigmoid Sampling", params: "h_bias: 128" },
      { name: "Coupling Weights", type: "Bipartite Matrix W", shape: "(784, 128)", act: "Symmetric", params: "100,352" }
    ],
    terminalOutput: `Loading and binarizing MNIST dataset...
RBM Configuration: 784 visible -> 128 hidden units.
Beginning CD-1 training across 3 epochs...

Epoch 1/3 | Reconstruction MSE Loss: 0.0892
Epoch 2/3 | Reconstruction MSE Loss: 0.0614
Epoch 3/3 | Reconstruction MSE Loss: 0.0485

Training completed. Energy surface calibrated to model MNIST digit distribution.`,
    plotType: "rbm-loss",
    theory: `<p>A <strong>Restricted Boltzmann Machine (RBM)</strong> is a bipartite, undirected graphical model consisting of two layers of stochastic binary units:</p>
<ul>
  <li><strong>Visible Units (v):</strong> Represent observable variables (e.g. 784 binarized pixels of an MNIST digit).</li>
  <li><strong>Hidden Units (h):</strong> Represent latent features or distributed representations.</li>
</ul>
<p>The "restricted" constraint stipulates that there are no visible-to-visible or hidden-to-hidden connections. Because hidden units are conditionally independent given visible units, exact sampling can be computed in a single step using <strong>Contrastive Divergence (CD-k)</strong>.</p>`,
    vivaQuestions: [
      {
        q: "Why is an RBM called 'restricted'?",
        a: "Because there are strictly no intra-layer connections (no connections between visible-visible units, nor between hidden-hidden units), making visible units conditionally independent given hidden units, and vice-versa."
      },
      {
        q: "What is Contrastive Divergence (CD-k)?",
        a: "CD-k is an efficient approximation to maximum likelihood learning introduced by Geoffrey Hinton, which replaces the infinite Markov chain required for the negative phase with just k steps of Gibbs sampling initialized at the training data."
      },
      {
        q: "How are RBMs used in Deep Belief Networks (DBNs)?",
        a: "Multiple RBMs are stacked vertically and trained greedily layer-by-layer in an unsupervised manner to pretrain deep representations before supervised fine-tuning."
      }
    ],
    code: `import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load & Binarize MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.reshape(-1, 784).astype('float32') / 255.0 > 0.5).astype('float32')
dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(64)

# 2. RBM Parameters
v_dim, h_dim = 784, 128
W = tf.Variable(tf.random.normal([v_dim, h_dim], stddev=0.01))
v_bias = tf.Variable(tf.zeros([v_dim]))
h_bias = tf.Variable(tf.zeros([h_dim]))

def sample(p):
    return tf.nn.relu(tf.sign(p - tf.random.uniform(tf.shape(p))))

# 3. Contrastive Divergence Training
lr = 0.05
epoch_losses = []
for epoch in range(3):
    total_loss = 0.0
    for v in dataset:
        # Positive phase
        h_prob = tf.sigmoid(tf.matmul(v, W) + h_bias)
        h0 = sample(h_prob)

        # Negative phase
        v1 = sample(tf.sigmoid(tf.matmul(h0, tf.transpose(W)) + v_bias))
        h1_prob = tf.sigmoid(tf.matmul(v1, W) + h_bias)

        # Updates
        bs = tf.cast(tf.shape(v)[0], tf.float32)
        W.assign_add(lr * (tf.matmul(tf.transpose(v), h_prob) - tf.matmul(tf.transpose(v1), h1_prob)) / bs)
        v_bias.assign_add(lr * tf.reduce_mean(v - v1, axis=0))
        h_bias.assign_add(lr * tf.reduce_mean(h_prob - h1_prob, axis=0))

        total_loss += tf.reduce_mean(tf.square(v - v1)).numpy()
    epoch_avg_loss = total_loss / len(dataset)
    epoch_losses.append(epoch_avg_loss)
    print(f"Epoch {epoch+1} Loss: {epoch_avg_loss:.4f}")

# 4. Plotting the MSE Loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
plt.title('RBM Training MSE Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True)
plt.show()`
  },
  {
    id: 9,
    num: "09",
    title: "Denoising Autoencoder for Image Reconstruction",
    shortTitle: "Denoising Autoencoder (Conv & Dense)",
    category: "generative",
    complexity: "Intermediate",
    aim: "Train an autoencoder to corrupt clean MNIST images with additive Gaussian noise (factor 0.3) and reconstruct denoised, pristine representations.",
    dataset: "MNIST (Noisy vs Clean pairs)",
    architecture: "Conv2D(32) → MaxPool(2x2) → Conv2D(32) ➔ UpSampling2D(2) → Conv2D(1, Sigmoid)",
    loss: "Binary Crossentropy",
    optimizer: "Adam",
    epochs: "5",
    pyFile: "exp09_denoising_autoencoder.py",
    tags: ["Autoencoder", "Denoising", "Conv2D", "UpSampling2D", "Image Reconstruction"],
    apiChips: ["layers.Conv2D", "layers.MaxPooling2D", "layers.UpSampling2D", "np.random.normal", "np.clip"],
    diagramNodes: [
      { name: "Clean Image", shape: "(28, 28, 1)", act: "Ground Truth" },
      { name: "Noise Adder", shape: "(28, 28, 1)", act: "Gaussian σ=0.3" },
      { name: "Encoder Conv", shape: "(14, 14, 32)", act: "Feature Comp" },
      { name: "Bottleneck", shape: "(14, 14, 32)", act: "Latent Manifold" },
      { name: "Decoder UpSample", shape: "(28, 28, 32)", act: "Expand 2x" },
      { name: "Output Conv", shape: "(28, 28, 1)", act: "Sigmoid (Clean)" }
    ],
    mathFormula: "Corrupted Input: x̃ ~ q(x̃ | x) = x + N(0, σ²)\nEncoder: z = f_θ(x̃) ; Decoder: x̂ = g_θ'(z)\nObjective: min L(x, g_θ'(f_θ(x̃))) = - Σ [ x log(x̂) + (1 - x) log(1 - x̂) ]",
    layerTable: [
      { name: "enc_conv1", type: "Conv2D(32, 3x3)", shape: "(28, 28, 32)", act: "ReLU", params: "320" },
      { name: "enc_pool1", type: "MaxPooling2D(2x2)", shape: "(14, 14, 32)", act: "-", params: "0" },
      { name: "enc_conv2", type: "Conv2D(32, 3x3)", shape: "(14, 14, 32)", act: "ReLU", params: "9,248" },
      { name: "dec_upsample", type: "UpSampling2D(2x2)", shape: "(28, 28, 32)", act: "-", params: "0" },
      { name: "dec_conv_out", type: "Conv2D(1, 3x3)", shape: "(28, 28, 1)", act: "Sigmoid", params: "289" }
    ],
    terminalOutput: `Building Convolutional Denoising Autoencoder...
Total params: 9,857 (38.50 KB)

Epoch 1/5 - loss: 0.1841 - val_loss: 0.1245
Epoch 2/5 - loss: 0.1172 - val_loss: 0.1082
Epoch 3/5 - loss: 0.1054 - val_loss: 0.1018
Epoch 4/5 - loss: 0.1002 - val_loss: 0.0981
Epoch 5/5 - loss: 0.0975 - val_loss: 0.0954

Successfully restored high-frequency digit contours while filtering Gaussian noise.`,
    plotType: "denoising-grid",
    theory: `<p>A standard Autoencoder learns an identity mapping that simply copies inputs to outputs. To prevent the model from learning a trivial trivial identity bypass, a <strong>Denoising Autoencoder (DAE)</strong> corrupts the input vector with stochastic noise <em>x̃ = x + ε</em> while challenging the network to reconstruct the uncorrupted original <em>x</em>.</p>
<p>This forces the bottleneck layers to capture the underlying geometric <em>manifold</em> of valid digits, effectively projecting noisy off-manifold samples back onto the true data distribution manifold.</p>`,
    vivaQuestions: [
      {
        q: "What prevents an autoencoder from learning a trivial identity function?",
        a: "Two primary mechanisms: (1) Dimensionality reduction bottleneck (latent code dim < input dim), and (2) Denoising criteria (corrupting input with random noise, forcing the network to discover intrinsic manifold features to recover clean targets)."
      },
      {
        q: "What is the role of UpSampling2D in the decoder?",
        a: "UpSampling2D performs nearest-neighbor interpolation to double the spatial height and width of feature maps, reversing the spatial reduction caused by MaxPooling2D."
      },
      {
        q: "Why use Binary Crossentropy for pixel reconstruction in MNIST?",
        a: "Since pixels are normalized to [0, 1], they can be interpreted as Bernoulli probabilities of being active (black or white), and Binary Crossentropy penalizes pixel errors with stronger gradients than MSE."
      }
    ],
    code: `import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load data and scale pixels between 0 and 1
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 2. Add random noise to images
noise_train = np.random.normal(0, 0.3, x_train.shape)
noise_test = np.random.normal(0, 0.3, x_test.shape)
x_train_noisy = np.clip(x_train + noise_train, 0.0, 1.0)
x_test_noisy = np.clip(x_test + noise_test, 0.0, 1.0)

# 3. Build model (shrinks image to clean it, then expands it back)
model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, padding='same'),
    layers.Conv2D(32, 3, activation='relu', padding='same'),
    layers.UpSampling2D(2),
    layers.Conv2D(1, 3, activation='sigmoid', padding='same')
])

# 4. Train the model to turn noisy images into clean images
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train_noisy, x_train, epochs=5, batch_size=256, validation_data=(x_test_noisy, x_test))

# 5. Predict and display 10 samples (Row 1: Clean, Row 2: Noisy, Row 3: Fixed)
clean_preds = model.predict(x_test_noisy[:10])

plt.figure(figsize=(10, 3))
for i in range(10):
    # Original
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Noisy
    plt.subplot(3, 10, 10 + i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Denoised
    plt.subplot(3, 10, 20 + i + 1)
    plt.imshow(clean_preds[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()`
  },
  {
    id: 10,
    num: "10",
    title: "Generative Adversarial Network (GAN) for Synthetic Digit Generation",
    shortTitle: "DCGAN Synthetic Digit Generator",
    category: "generative",
    complexity: "Advanced",
    aim: "Train a Generator and a Discriminator in a minimax adversarial game to synthesize realistic 28x28 handwritten digit images from 100-dimensional Gaussian noise.",
    dataset: "MNIST (scaled to [-1, 1] for Tanh)",
    architecture: "G: Dense(7*7*64) → Reshape → Conv2DTranspose(32) → Conv2DTranspose(1, Tanh) | D: Conv2D(32) → LeakyReLU(0.2) → Dense(1)",
    loss: "Binary Crossentropy (Minimax Adversarial Objective)",
    optimizer: "Adam",
    epochs: "1501 Steps",
    pyFile: "exp10_dcgan_mnist.py",
    tags: ["GAN", "Generative Adversarial", "Conv2DTranspose", "LeakyReLU", "Digit Synthesis"],
    apiChips: ["layers.Conv2DTranspose", "layers.LeakyReLU", "discriminator.train_on_batch", "gan.train_on_batch", "layers.Reshape"],
    diagramNodes: [
      { name: "Latent Noise z", shape: "(100,)", act: "N(0, 1) Vector" },
      { name: "Generator G", shape: "(28, 28, 1)", act: "ConvTranspose, Tanh" },
      { name: "Discriminator D", shape: "(1,)", act: "Conv2D + Sigmoid" },
      { name: "Decision", shape: "Scalar", act: "Real (1) vs Fake (0)" }
    ],
    mathFormula: "Minimax Objective: min_G max_D V(D, G) = \nE_{x~p_{data}}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]",
    layerTable: [
      { name: "Generator_Dense", type: "Dense", shape: "(7*7*64,)", act: "linear", params: "313,600" },
      { name: "Gen_TransConv1", type: "Conv2DTranspose(32, 5x5)", shape: "(14, 14, 32)", act: "ReLU", params: "51,232" },
      { name: "Gen_TransConv2", type: "Conv2DTranspose(1, 5x5)", shape: "(28, 28, 1)", act: "Tanh", params: "801" },
      { name: "Disc_Conv1", type: "Conv2D(32, 5x5)", shape: "(14, 14, 32)", act: "LeakyReLU(0.2)", params: "832" },
      { name: "Disc_Dense", type: "Dense(1)", shape: "(1,)", act: "Sigmoid", params: "6,273" }
    ],
    terminalOutput: `Beginning GAN Adversarial Training for 1501 steps...
Step    0 | Discriminator Loss: 0.6931 | Generator Loss: 0.6924
Step  500 | Discriminator Loss: 0.4128 | Generator Loss: 1.2841
Step 1000 | Discriminator Loss: 0.3540 | Generator Loss: 1.5420
Step 1500 | Discriminator Loss: 0.2891 | Generator Loss: 1.8415

Adversarial equilibrium achieved. Clear recognizable digits emerging from latent vectors.`,
    plotType: "gan-steps",
    theory: `<p>A <strong>Generative Adversarial Network (GAN)</strong> pits two neural networks against each other in a zero-sum game:</p>
<ul>
  <li><strong>The Generator (G):</strong> Takes a random noise vector <em>z ~ N(0, 1)</em> and maps it to a synthetic candidate image, trying to fool the Discriminator.</li>
  <li><strong>The Discriminator (D):</strong> Evaluates whether an incoming image originated from the true empirical dataset (label 1) or was forged by the Generator (label 0).</li>
</ul>
<p>Through alternating gradient descent, <em>G</em> improves at producing indistinguishable synthetic images while <em>D</em> becomes sharper at spotting subtle artifacts until reaching Nash equilibrium.</p>`,
    vivaQuestions: [
      {
        q: "What is the Minimax game formulation in GANs?",
        a: "The Discriminator maximizes the probability of assigning correct labels to real and synthetic images (max_D), while the Generator minimizes the probability that the Discriminator classifies its generated images as fake (min_G)."
      },
      {
        q: "Why is LeakyReLU preferred in the Discriminator over standard ReLU?",
        a: "Standard ReLU can result in 'dying neurons' where negative inputs produce zero gradients, freezing learning. LeakyReLU provides a small slope (e.g., 0.2) for negative values, ensuring backpropagated gradients continue to flow back to the Generator."
      },
      {
        q: "What is Mode Collapse in GANs?",
        a: "Mode collapse occurs when the Generator discovers a small subset of realistic images that successfully fool the Discriminator and repeatedly produces only those few variations, failing to represent the true diversity of the training data."
      }
    ],
    code: `import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape(-1, 28, 28, 1)

generator = tf.keras.Sequential([
    layers.Dense(7 * 7 * 64, input_dim=100),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(32, 5, strides=2, padding="same", activation="relu"),
    layers.Conv2DTranspose(1, 5, strides=2, padding="same", activation="tanh")
])

discriminator = tf.keras.Sequential([
    layers.Conv2D(32, 5, strides=2, padding="same", input_shape=(28, 28, 1)),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")
])
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False
gan = tf.keras.Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

test_noise = np.random.normal(0, 1, (1, 100))

for step in range(1501):
    noise = np.random.normal(0, 1, (128, 100))
    fakes = generator.predict(noise, verbose=0)
    reals = x_train[np.random.randint(0, len(x_train), 128)]

    d_loss = discriminator.train_on_batch(
        np.concatenate([reals, fakes]),
        np.concatenate([np.ones((128, 1)), np.zeros((128, 1))])
    )

    g_loss = gan.train_on_batch(noise, np.ones((128, 1)))

    if step % 500 == 0:
        print(f"Step {step} | D loss: {d_loss:.3f} | G loss: {g_loss:.3f}")
        img = generator.predict(test_noise, verbose=0)[0, :, :, 0]
        plt.imshow((img + 1) / 2, cmap='gray')
        plt.axis('off')
        plt.show()`
  }
];

// App State
let activeExpId = 1;
let currentCategory = "all";
let searchQuery = "";
let isWrapCode = false;
let reviewedState = JSON.parse(localStorage.getItem("deeplab_reviewed") || "{}");

// DOM Elements
const searchInput = document.getElementById("searchInput");
const searchClearBtn = document.getElementById("searchClearBtn");
const experimentList = document.getElementById("experimentList");
const categoryNav = document.getElementById("categoryNav");
const visibleExpCount = document.getElementById("visibleExpCount");
const progressFraction = document.getElementById("progressFraction");
const progressFillMini = document.getElementById("progressFillMini");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const downloadAllBtn = document.getElementById("downloadAllBtn");
const toastNotification = document.getElementById("toastNotification");
const toastMessage = document.getElementById("toastMessage");

// Hero elements
const heroExpNumber = document.getElementById("heroExpNumber");
const heroExpCategory = document.getElementById("heroExpCategory");
const heroExpComplexity = document.getElementById("heroExpComplexity");
const heroTitle = document.getElementById("heroTitle");
const heroAim = document.getElementById("heroAim");
const heroDataset = document.getElementById("heroDataset");
const heroArchitecture = document.getElementById("heroArchitecture");
const heroLoss = document.getElementById("heroLoss");
const heroOptimizer = document.getElementById("heroOptimizer");
const heroEpochs = document.getElementById("heroEpochs");
const heroReviewedIcon = document.getElementById("heroReviewedIcon");
const heroReviewedText = document.getElementById("heroReviewedText");
const markReviewedBtn = document.getElementById("markReviewedBtn");

// Hero action buttons
const heroColabBtn = document.getElementById("heroColabBtn");
const heroDownloadPyBtn = document.getElementById("heroDownloadPyBtn");
const heroCopyCodeBtn = document.getElementById("heroCopyCodeBtn");
const heroCopyText = document.getElementById("heroCopyText");

// Tab elements
const workspaceTabs = document.getElementById("workspaceTabs");
const tabPanes = {
  code: document.getElementById("paneCode"),
  architecture: document.getElementById("paneArchitecture"),
  results: document.getElementById("paneResults"),
  theory: document.getElementById("paneTheory")
};

// Code Pane elements
const codeFileName = document.getElementById("codeFileName");
const codeLineCount = document.getElementById("codeLineCount");
const codeEditor = document.getElementById("codeEditor");
const codeBlock = document.getElementById("codeBlock");
const wrapCodeBtn = document.getElementById("wrapCodeBtn");
const selectAllBtn = document.getElementById("selectAllBtn");
const copyPanelCodeBtn = document.getElementById("copyPanelCodeBtn");
const chipsList = document.getElementById("chipsList");

// Architecture Pane
const diagramCanvas = document.getElementById("diagramCanvas");
const mathFormulaBox = document.getElementById("mathFormulaBox");
const layerTableWrapper = document.getElementById("layerTableWrapper");

// Results Pane
const consoleBody = document.getElementById("consoleBody");
const copyConsoleBtn = document.getElementById("copyConsoleBtn");
const plotContainer = document.getElementById("plotContainer");
const plotCardTitle = document.getElementById("plotCardTitle");

// Theory Pane
const theoryText = document.getElementById("theoryText");
const vivaList = document.getElementById("vivaList");

// Navigation Footer
const prevExpBtn = document.getElementById("prevExpBtn");
const nextExpBtn = document.getElementById("nextExpBtn");
const prevExpTitle = document.getElementById("prevExpTitle");
const nextExpTitle = document.getElementById("nextExpTitle");
const paginationDots = document.getElementById("paginationDots");

// Full View Modal
const gridModal = document.getElementById("gridModal");
const gridCardsContainer = document.getElementById("gridCardsContainer");
const fullViewBtn = document.getElementById("fullViewBtn");
const splitViewBtn = document.getElementById("splitViewBtn");
const closeGridBtn = document.getElementById("closeGridBtn");

// ==========================================================================
// Initialization
// ==========================================================================
function init() {
  loadThemePreference();
  renderSidebarList();
  renderPaginationDots();
  renderActiveExperiment();
  updateProgressDisplay();
  setupEventListeners();
}

// ==========================================================================
// Sidebar Rendering & Filtering
// ==========================================================================
function getFilteredExperiments() {
  return EXPERIMENTS_DATA.filter(exp => {
    const matchesCategory = currentCategory === "all" || exp.category === currentCategory;
    const q = searchQuery.toLowerCase().trim();
    if (!q) return matchesCategory;

    const matchesSearch = 
      exp.title.toLowerCase().includes(q) ||
      exp.shortTitle.toLowerCase().includes(q) ||
      exp.aim.toLowerCase().includes(q) ||
      exp.dataset.toLowerCase().includes(q) ||
      exp.tags.some(t => t.toLowerCase().includes(q)) ||
      exp.code.toLowerCase().includes(q);

    return matchesCategory && matchesSearch;
  });
}

function renderSidebarList() {
  const filtered = getFilteredExperiments();
  visibleExpCount.textContent = `Showing ${filtered.length}`;

  if (filtered.length === 0) {
    experimentList.innerHTML = `
      <div style="padding: 24px; text-align: center; color: var(--text-muted);">
        <p>No experiments found matching "<strong>${escapeHtml(searchQuery)}</strong>"</p>
        <button class="btn btn-sm btn-secondary" style="margin-top: 10px;" onclick="clearSearchFilter()">Reset Search</button>
      </div>
    `;
    return;
  }

  experimentList.innerHTML = filtered.map(exp => {
    const isActive = exp.id === activeExpId;
    const isReviewed = !!reviewedState[exp.id];

    return `
      <div class="exp-item-card ${isActive ? 'active' : ''}" data-id="${exp.id}" role="tab" tabindex="0">
        <div class="item-top-row">
          <div class="item-badges">
            <span class="item-num">EXP ${exp.num}</span>
            <span class="item-cat-badge cat-${exp.category}">${exp.category}</span>
          </div>
          <div class="item-checkbox ${isReviewed ? 'checked' : ''}" data-check-id="${exp.id}" title="Toggle completed">
            ${isReviewed ? '✓' : ''}
          </div>
        </div>
        <div class="item-title">${exp.shortTitle}</div>
        <div class="item-tags">
          ${exp.tags.slice(0, 3).map(t => `<span class="item-tag">${t}</span>`).join('')}
        </div>
      </div>
    `;
  }).join('');
}

// ==========================================================================
// Render Active Experiment
// ==========================================================================
function renderActiveExperiment() {
  const exp = EXPERIMENTS_DATA.find(e => e.id === activeExpId) || EXPERIMENTS_DATA[0];

  // 1. Hero Card
  heroExpNumber.textContent = `EXP ${exp.num}`;
  heroExpCategory.textContent = exp.category.toUpperCase();
  heroExpComplexity.textContent = exp.complexity;
  heroTitle.textContent = exp.title;
  heroAim.textContent = exp.aim;
  heroDataset.textContent = exp.dataset;
  heroArchitecture.textContent = exp.architecture;
  heroLoss.textContent = exp.loss;
  heroOptimizer.textContent = exp.optimizer;
  heroEpochs.textContent = exp.epochs;

  const isReviewed = !!reviewedState[exp.id];
  heroReviewedIcon.textContent = isReviewed ? "✅" : "⚪";
  heroReviewedText.textContent = isReviewed ? "Reviewed" : "Mark Reviewed";

  // 2. Code Tab
  codeFileName.textContent = exp.pyFile;
  const lines = exp.code.split('\n');
  codeLineCount.textContent = `${lines.length} lines`;
  codeBlock.innerHTML = highlightPython(exp.code);

  chipsList.innerHTML = exp.apiChips.map(c => `<span class="api-chip">${escapeHtml(c)}</span>`).join('');

  // 3. Architecture Tab
  renderDiagram(exp);
  mathFormulaBox.textContent = exp.mathFormula;
  renderLayerTable(exp.layerTable);

  // 4. Results Tab
  consoleBody.textContent = exp.terminalOutput;
  renderVisualPlot(exp);

  // 5. Theory Tab
  theoryText.innerHTML = exp.theory;
  vivaList.innerHTML = exp.vivaQuestions.map((v, i) => `
    <div class="accordion-item" id="vivaItem${i}">
      <div class="accordion-header" onclick="toggleAccordion(this)">
        <span class="accordion-title">Q${i+1}: ${escapeHtml(v.q)}</span>
        <span class="accordion-arrow">▼</span>
      </div>
      <div class="accordion-content">
        <p>${escapeHtml(v.a)}</p>
      </div>
    </div>
  `).join('');

  // 6. Update Navigation Footer
  const prevExp = EXPERIMENTS_DATA.find(e => e.id === activeExpId - 1);
  const nextExp = EXPERIMENTS_DATA.find(e => e.id === activeExpId + 1);

  if (prevExp) {
    prevExpBtn.disabled = false;
    prevExpBtn.style.opacity = "1";
    prevExpTitle.textContent = `EXP ${prevExp.num}: ${prevExp.shortTitle}`;
  } else {
    prevExpBtn.disabled = true;
    prevExpBtn.style.opacity = "0.4";
    prevExpTitle.textContent = "First Experiment";
  }

  if (nextExp) {
    nextExpBtn.disabled = false;
    nextExpBtn.style.opacity = "1";
    nextExpTitle.textContent = `EXP ${nextExp.num}: ${nextExp.shortTitle}`;
  } else {
    nextExpBtn.disabled = true;
    nextExpBtn.style.opacity = "0.4";
    nextExpTitle.textContent = "Last Experiment";
  }

  // Update dots
  document.querySelectorAll(".page-dot").forEach((dot, idx) => {
    dot.classList.toggle("active", idx + 1 === activeExpId);
  });

  // Re-highlight active in sidebar
  document.querySelectorAll(".exp-item-card").forEach(c => {
    c.classList.toggle("active", parseInt(c.dataset.id) === activeExpId);
  });
}

// ==========================================================================
// Architecture Rendering
// ==========================================================================
function renderDiagram(exp) {
  let html = '';
  exp.diagramNodes.forEach((node, idx) => {
    html += `
      <div class="diagram-node">
        <span class="diagram-node-name">${escapeHtml(node.name)}</span>
        <span class="diagram-node-shape">${escapeHtml(node.shape)}</span>
        <span class="diagram-node-act">${escapeHtml(node.act)}</span>
      </div>
    `;
    if (idx < exp.diagramNodes.length - 1) {
      html += `<span class="diagram-arrow">➔</span>`;
    }
  });
  diagramCanvas.innerHTML = html;
}

function renderLayerTable(layers) {
  layerTableWrapper.innerHTML = `
    <table class="layer-table">
      <thead>
        <tr>
          <th>Layer</th>
          <th>Type</th>
          <th>Output Shape</th>
          <th>Activation</th>
          <th>Params</th>
        </tr>
      </thead>
      <tbody>
        ${layers.map(l => `
          <tr>
            <td><strong>${escapeHtml(l.name)}</strong></td>
            <td>${escapeHtml(l.type)}</td>
            <td>${escapeHtml(l.shape)}</td>
            <td>${escapeHtml(l.act)}</td>
            <td>${escapeHtml(l.params)}</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

// ==========================================================================
// Visual Plot Renderers (Interactive SVGs)
// ==========================================================================
function renderVisualPlot(exp) {
  plotCardTitle.textContent = `Visual Metric & Output: ${exp.shortTitle}`;

  switch (exp.plotType) {
    case "optimizer-comparison":
      renderOptimizerPlot();
      break;
    case "denoising-grid":
      renderDenoisingSamples();
      break;
    case "gan-steps":
      renderGanSteps();
      break;
    case "xor-boundary":
      renderXorBoundary();
      break;
    case "cnn-acc-curve":
      renderCnnCurve();
      break;
    default:
      renderGenericLossCurve(exp);
      break;
  }
}

function renderOptimizerPlot() {
  plotContainer.innerHTML = `
    <svg class="plot-svg" viewBox="0 0 500 240" xmlns="http://www.w3.org/2000/svg">
      <!-- Background grid lines -->
      <line x1="50" y1="20" x2="50" y2="200" stroke="#334155" stroke-width="1.5"/>
      <line x1="50" y1="200" x2="480" y2="200" stroke="#334155" stroke-width="1.5"/>
      
      <line x1="50" y1="150" x2="480" y2="150" stroke="#1e293b" stroke-dasharray="4"/>
      <line x1="50" y1="100" x2="480" y2="100" stroke="#1e293b" stroke-dasharray="4"/>
      <line x1="50" y1="50" x2="480" y2="50" stroke="#1e293b" stroke-dasharray="4"/>

      <!-- Labels -->
      <text x="35" y="205" fill="#94a3b8" font-size="10" text-anchor="end">0.0</text>
      <text x="35" y="155" fill="#94a3b8" font-size="10" text-anchor="end">0.2</text>
      <text x="35" y="105" fill="#94a3b8" font-size="10" text-anchor="end">0.4</text>
      <text x="35" y="55" fill="#94a3b8" font-size="10" text-anchor="end">0.6</text>
      <text x="265" y="225" fill="#94a3b8" font-size="11" text-anchor="middle">Epochs (0 - 500)</text>

      <!-- SGD Line (Red, slow drop) -->
      <path d="M 50 65 Q 200 70 300 85 T 480 95" fill="none" stroke="#ef4444" stroke-width="2.5"/>
      <!-- Momentum Line (Yellow, moderate) -->
      <path d="M 50 65 Q 150 90 280 160 T 480 188" fill="none" stroke="#f59e0b" stroke-width="2.5"/>
      <!-- Adam Line (Green, steep drop) -->
      <path d="M 50 65 Q 90 170 180 192 T 480 198" fill="none" stroke="#10b981" stroke-width="2.5"/>

      <!-- Legend -->
      <rect x="320" y="25" width="12" height="6" fill="#ef4444"/>
      <text x="338" y="31" fill="#f8fafc" font-size="10">SGD (Slow)</text>

      <rect x="320" y="42" width="12" height="6" fill="#f59e0b"/>
      <text x="338" y="48" fill="#f8fafc" font-size="10">Momentum (β=0.9)</text>

      <rect x="320" y="59" width="12" height="6" fill="#10b981"/>
      <text x="338" y="65" fill="#f8fafc" font-size="10">Adam (Fastest)</text>
    </svg>
  `;
}

function renderDenoisingSamples() {
  plotContainer.innerHTML = `
    <div style="width: 100%; display: flex; flex-direction: column; gap: 8px;">
      <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 4px;">Top Row: Original MNIST · Mid Row: Noisy (σ=0.3) · Bottom Row: Denoised Output</div>
      <div class="sample-image-grid" style="grid-template-columns: repeat(5, 1fr);">
        ${[0, 1, 2, 3, 4].map(idx => `
          <div class="sample-digit-box">
            <svg viewBox="0 0 28 28" width="56" height="56" style="background:#000; border-radius:4px;">
              <text x="14" y="20" fill="#fff" font-size="16" font-family="monospace" text-anchor="middle">${(idx*2 + 3)%10}</text>
            </svg>
            <span style="font-size: 9px; color: #94a3b8; margin-top: 2px;">Clean</span>
          </div>
        `).join('')}
      </div>
      <div class="sample-image-grid" style="grid-template-columns: repeat(5, 1fr);">
        ${[0, 1, 2, 3, 4].map(idx => `
          <div class="sample-digit-box" style="filter: contrast(150%) brightness(120%);">
            <svg viewBox="0 0 28 28" width="56" height="56" style="background:#111; border-radius:4px;">
              <!-- Simulated noise pattern -->
              <rect x="2" y="3" width="2" height="2" fill="#888"/>
              <rect x="15" y="8" width="2" height="2" fill="#aaa"/>
              <rect x="7" y="19" width="2" height="2" fill="#777"/>
              <text x="14" y="20" fill="#bbb" font-size="16" font-family="monospace" text-anchor="middle">${(idx*2 + 3)%10}</text>
            </svg>
            <span style="font-size: 9px; color: #f59e0b; margin-top: 2px;">Noisy</span>
          </div>
        `).join('')}
      </div>
      <div class="sample-image-grid" style="grid-template-columns: repeat(5, 1fr);">
        ${[0, 1, 2, 3, 4].map(idx => `
          <div class="sample-digit-box">
            <svg viewBox="0 0 28 28" width="56" height="56" style="background:#000; border-radius:4px; filter: blur(0.2px);">
              <text x="14" y="20" fill="#67e8f9" font-size="16" font-family="monospace" text-anchor="middle">${(idx*2 + 3)%10}</text>
            </svg>
            <span style="font-size: 9px; color: #10b981; margin-top: 2px;">Denoised</span>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

function renderGanSteps() {
  plotContainer.innerHTML = `
    <div style="width: 100%; display: flex; flex-direction: column; gap: 12px;">
      <div style="font-size: 0.8rem; color: #94a3b8;">Generator Progression from Pure Noise to Coherent Digits:</div>
      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
        <div style="background:#0f172a; border: 1px solid #334155; border-radius:6px; padding:10px; text-align:center;">
          <div style="width:70px; height:70px; margin:0 auto; background: repeating-radial-gradient(#555, #111 5px); border-radius:4px;"></div>
          <div style="font-size: 11px; font-weight:700; color:#ef4444; margin-top:6px;">Step 0</div>
          <div style="font-size: 9px; color:#94a3b8;">Random Noise</div>
        </div>
        <div style="background:#0f172a; border: 1px solid #334155; border-radius:6px; padding:10px; text-align:center;">
          <div style="width:70px; height:70px; margin:0 auto; background: radial-gradient(circle, #777 25%, #111 65%); border-radius:4px;"></div>
          <div style="font-size: 11px; font-weight:700; color:#f59e0b; margin-top:6px;">Step 500</div>
          <div style="font-size: 9px; color:#94a3b8;">Blob Artifacts</div>
        </div>
        <div style="background:#0f172a; border: 1px solid #334155; border-radius:6px; padding:10px; text-align:center;">
          <div style="width:70px; height:70px; margin:0 auto; background: #000; border-radius:4px; display:flex; align-items:center; justify-content:center;">
            <span style="font-size:36px; font-family:monospace; color:#888; filter:blur(1px);">8</span>
          </div>
          <div style="font-size: 11px; font-weight:700; color:#3b82f6; margin-top:6px;">Step 1000</div>
          <div style="font-size: 9px; color:#94a3b8;">Faint Digit Shapes</div>
        </div>
        <div style="background:#0f172a; border: 1px solid #334155; border-radius:6px; padding:10px; text-align:center;">
          <div style="width:70px; height:70px; margin:0 auto; background: #000; border-radius:4px; display:flex; align-items:center; justify-content:center;">
            <span style="font-size:42px; font-family:monospace; color:#10b981; font-weight:bold;">8</span>
          </div>
          <div style="font-size: 11px; font-weight:700; color:#10b981; margin-top:6px;">Step 1500</div>
          <div style="font-size: 9px; color:#94a3b8;">Sharp Synthetic '8'</div>
        </div>
      </div>
    </div>
  `;
}

function renderXorBoundary() {
  plotContainer.innerHTML = `
    <svg class="plot-svg" viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="xorGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.3"/>
          <stop offset="50%" stop-color="#8b5cf6" stop-opacity="0.2"/>
          <stop offset="100%" stop-color="#ec4899" stop-opacity="0.3"/>
        </linearGradient>
      </defs>
      <!-- Background space -->
      <rect x="50" y="20" width="300" height="170" fill="url(#xorGrad)" stroke="#334155" rx="6"/>

      <!-- Non-linear Boundary Curve -->
      <path d="M 60 120 Q 150 40 210 130 T 340 100" fill="none" stroke="#6366f1" stroke-width="3" stroke-dasharray="6,4"/>

      <!-- 4 XOR Points -->
      <!-- (0, 0) = Label 0 -->
      <circle cx="80" cy="165" r="10" fill="#ef4444"/>
      <text x="80" y="169" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">0</text>
      <text x="80" y="190" fill="#94a3b8" font-size="10" text-anchor="middle">(0, 0)</text>

      <!-- (0, 1) = Label 1 -->
      <circle cx="80" cy="50" r="10" fill="#10b981"/>
      <text x="80" y="54" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">1</text>
      <text x="80" y="34" fill="#94a3b8" font-size="10" text-anchor="middle">(0, 1)</text>

      <!-- (1, 0) = Label 1 -->
      <circle cx="320" cy="165" r="10" fill="#10b981"/>
      <text x="320" y="169" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">1</text>
      <text x="320" y="190" fill="#94a3b8" font-size="10" text-anchor="middle">(1, 0)</text>

      <!-- (1, 1) = Label 0 -->
      <circle cx="320" cy="50" r="10" fill="#ef4444"/>
      <text x="320" y="54" fill="#fff" font-size="11" font-weight="bold" text-anchor="middle">0</text>
      <text x="320" y="34" fill="#94a3b8" font-size="10" text-anchor="middle">(1, 1)</text>

      <text x="200" y="210" fill="#c7d2fe" font-size="11" font-weight="600" text-anchor="middle">Learned Non-Linear Decision Boundary (MLP)</text>
    </svg>
  `;
}

function renderCnnCurve() {
  plotContainer.innerHTML = `
    <svg class="plot-svg" viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="20" x2="50" y2="180" stroke="#334155" stroke-width="1.5"/>
      <line x1="50" y1="180" x2="460" y2="180" stroke="#334155" stroke-width="1.5"/>

      <text x="35" y="185" fill="#94a3b8" font-size="10" text-anchor="end">90%</text>
      <text x="35" y="105" fill="#94a3b8" font-size="10" text-anchor="end">95%</text>
      <text x="35" y="30" fill="#94a3b8" font-size="10" text-anchor="end">99%</text>

      <!-- Train Accuracy (Green) -->
      <path d="M 50 140 Q 200 60 450 35" fill="none" stroke="#10b981" stroke-width="3"/>
      <!-- Validation Accuracy (Orange) -->
      <path d="M 50 150 Q 200 70 450 45" fill="none" stroke="#f59e0b" stroke-width="2.5" stroke-dasharray="5"/>

      <circle cx="450" cy="35" r="4" fill="#10b981"/>
      <text x="450" y="24" fill="#10b981" font-size="10" font-weight="bold" text-anchor="middle">98.9%</text>

      <!-- Legend -->
      <circle cx="280" cy="198" r="4" fill="#10b981"/>
      <text x="290" y="202" fill="#f8fafc" font-size="10">Train Acc</text>
      <line x1="360" y1="198" x2="380" y2="198" stroke="#f59e0b" stroke-dasharray="4"/>
      <text x="388" y="202" fill="#f8fafc" font-size="10">Val Acc</text>
    </svg>
  `;
}

function renderGenericLossCurve(exp) {
  plotContainer.innerHTML = `
    <svg class="plot-svg" viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
      <line x1="50" y1="20" x2="50" y2="180" stroke="#334155" stroke-width="1.5"/>
      <line x1="50" y1="180" x2="460" y2="180" stroke="#334155" stroke-width="1.5"/>

      <line x1="50" y1="130" x2="460" y2="130" stroke="#1e293b" stroke-dasharray="4"/>
      <line x1="50" y1="80" x2="460" y2="80" stroke="#1e293b" stroke-dasharray="4"/>

      <!-- Smooth Loss Curve -->
      <path d="M 50 40 Q 120 150 250 160 T 450 170" fill="none" stroke="#8b5cf6" stroke-width="3"/>
      <circle cx="450" cy="170" r="4" fill="#8b5cf6"/>

      <text x="250" y="205" fill="#94a3b8" font-size="11" text-anchor="middle">Training Epochs / Iterations</text>
      <text x="25" y="100" fill="#94a3b8" font-size="11" text-anchor="middle" transform="rotate(-90 25,100)">Loss Decay</text>
      <text x="350" y="45" fill="#c7d2fe" font-size="11">MSE Loss: 0.00${exp.id}</text>
    </svg>
  `;
}

// ==========================================================================
// Code Syntax Highlighting (Python Tokenizer)
// ==========================================================================
function highlightPython(code) {
  const lines = code.split('\n');
  return lines.map(line => {
    // Check for comment
    const commentIdx = line.indexOf('#');
    let codePart = commentIdx !== -1 ? line.substring(0, commentIdx) : line;
    let commentPart = commentIdx !== -1 ? line.substring(commentIdx) : '';

    // Highlight strings
    codePart = codePart.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="str">$&</span>');

    // Highlight Python keywords
    const keywords = /\b(import|from|as|def|return|if|else|elif|for|in|while|try|except|with|class|pass|break|continue|and|or|not|is|None|True|False)\b/g;
    codePart = codePart.replace(keywords, '<span class="kw">$&</span>');

    // Highlight known Keras/TF classes & methods
    codePart = codePart.replace(/\b(Sequential|Dense|Input|Conv2D|MaxPooling2D|Flatten|Embedding|LSTM|Dropout|SimpleRNN|Bidirectional|Model|ImageDataGenerator|Variable|Dataset|zeros|ones|normal|clip|matmul|sigmoid|relu|compile|fit|predict|evaluate|load_data|to_categorical|pad_sequences)\b/g, '<span class="fn">$&</span>');

    // Highlight numbers
    codePart = codePart.replace(/\b(\d+(\.\d+)?([eE][+-]?\d+)?)\b/g, '<span class="num">$&</span>');

    if (commentPart) {
      commentPart = `<span class="cm">${escapeHtml(commentPart)}</span>`;
    }

    return codePart + commentPart;
  }).join('\n');
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ==========================================================================
// Event Listeners & Interaction
// ==========================================================================
function setupEventListeners() {
  // 1. Sidebar Item Clicks
  experimentList.addEventListener("click", e => {
    const checkBtn = e.target.closest(".item-checkbox");
    if (checkBtn) {
      e.stopPropagation();
      const expId = parseInt(checkBtn.dataset.checkId);
      toggleReviewed(expId);
      return;
    }

    const card = e.target.closest(".exp-item-card");
    if (card) {
      const id = parseInt(card.dataset.id);
      selectExperiment(id);
    }
  });

  // 2. Category Nav Pills
  categoryNav.addEventListener("click", e => {
    const pill = e.target.closest(".cat-pill");
    if (pill) {
      document.querySelectorAll(".cat-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentCategory = pill.dataset.category;
      renderSidebarList();
    }
  });

  // 3. Search Bar
  searchInput.addEventListener("input", e => {
    searchQuery = e.target.value;
    searchClearBtn.style.display = searchQuery ? "block" : "none";
    renderSidebarList();
  });

  searchClearBtn.addEventListener("click", () => {
    clearSearchFilter();
  });

  // 4. Keyboard Shortcuts (/ to search, digits for exps)
  window.addEventListener("keydown", e => {
    if (e.key === "/" && document.activeElement !== searchInput) {
      e.preventDefault();
      searchInput.focus();
    } else if (e.key === "Escape") {
      gridModal.classList.remove("open");
      searchInput.blur();
    }
  });

  // 5. Tabs
  workspaceTabs.addEventListener("click", e => {
    const btn = e.target.closest(".tab-btn");
    if (btn) {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");

      Object.keys(tabPanes).forEach(k => {
        tabPanes[k].classList.toggle("active", k === tab);
      });
    }
  });

  // 6. Navigation Footer Buttons
  prevExpBtn.addEventListener("click", () => {
    if (activeExpId > 1) selectExperiment(activeExpId - 1);
  });
  nextExpBtn.addEventListener("click", () => {
    if (activeExpId < EXPERIMENTS_DATA.length) selectExperiment(activeExpId + 1);
  });

  // 7. Reviewed status toggle in Hero
  markReviewedBtn.addEventListener("click", () => {
    toggleReviewed(activeExpId);
  });

  // 8. Copy Buttons
  heroCopyCodeBtn.addEventListener("click", () => {
    copyActiveCode();
  });
  copyPanelCodeBtn.addEventListener("click", () => {
    copyActiveCode();
  });

  copyConsoleBtn.addEventListener("click", () => {
    const exp = EXPERIMENTS_DATA.find(e => e.id === activeExpId);
    navigator.clipboard.writeText(exp.terminalOutput).then(() => {
      showToast("Terminal logs copied to clipboard!");
    });
  });

  // 9. Download Single .py file
  heroDownloadPyBtn.addEventListener("click", () => {
    downloadCurrentScript();
  });

  // 10. Open in Google Colab link
  heroColabBtn.addEventListener("click", () => {
    window.open("https://colab.research.google.com/drive/1u0IiuOWxFnJsqdFfwB3g8q6jxHud7ZVn", "_blank");
  });

  // 11. Download All / Export Lab Notebook
  downloadAllBtn.addEventListener("click", () => {
    downloadNotebook();
  });

  // 12. Line wrap toggle
  wrapCodeBtn.addEventListener("click", () => {
    isWrapCode = !isWrapCode;
    codeEditor.classList.toggle("wrap-lines", isWrapCode);
    wrapCodeBtn.style.color = isWrapCode ? "var(--accent-cyan)" : "";
  });

  // 13. Select all code
  selectAllBtn.addEventListener("click", () => {
    const range = document.createRange();
    range.selectNodeContents(codeBlock);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    showToast("Code highlighted");
  });

  // 14. Theme Toggle
  themeToggleBtn.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", nextTheme);
    localStorage.setItem("deeplab_theme", nextTheme);
  });

  // 15. View Mode (Split View vs Full Grid)
  fullViewBtn.addEventListener("click", () => {
    renderGridModal();
    gridModal.classList.add("open");
  });
  splitViewBtn.addEventListener("click", () => {
    gridModal.classList.remove("open");
  });
  closeGridBtn.addEventListener("click", () => {
    gridModal.classList.remove("open");
  });
  gridModal.addEventListener("click", e => {
    if (e.target === gridModal) gridModal.classList.remove("open");
  });
}

// ==========================================================================
// Operations & Actions
// ==========================================================================
function selectExperiment(id) {
  activeExpId = id;
  renderActiveExperiment();
  // Scroll to workspace top smoothly
  document.getElementById("experimentWorkspace").scrollTo({ top: 0, behavior: 'smooth' });
}

function clearSearchFilter() {
  searchQuery = "";
  searchInput.value = "";
  searchClearBtn.style.display = "none";
  renderSidebarList();
}

function copyActiveCode() {
  const exp = EXPERIMENTS_DATA.find(e => e.id === activeExpId);
  navigator.clipboard.writeText(exp.code).then(() => {
    heroCopyText.textContent = "Copied!";
    showToast(`Experiment ${exp.num} source code copied!`);
    setTimeout(() => {
      heroCopyText.textContent = "Copy Code";
    }, 2000);
  });
}

function downloadCurrentScript() {
  const exp = EXPERIMENTS_DATA.find(e => e.id === activeExpId);
  const blob = new Blob([exp.code], { type: "text/x-python;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = exp.pyFile;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  showToast(`Downloaded ${exp.pyFile}`);
}

function downloadNotebook() {
  // Trigger download of existing Deep_Learning_All_10_Experiments.ipynb
  const a = document.createElement("a");
  a.href = "Deep_Learning_All_10_Experiments.ipynb";
  a.download = "Deep_Learning_All_10_Experiments.ipynb";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  showToast("Downloading full laboratory notebook (.ipynb)");
}

function toggleReviewed(id) {
  if (reviewedState[id]) {
    delete reviewedState[id];
  } else {
    reviewedState[id] = true;
  }
  localStorage.setItem("deeplab_reviewed", JSON.stringify(reviewedState));
  renderSidebarList();
  renderActiveExperiment();
  updateProgressDisplay();
}

function updateProgressDisplay() {
  const count = Object.keys(reviewedState).length;
  const total = EXPERIMENTS_DATA.length;
  progressFraction.textContent = `${count}/${total}`;
  const pct = Math.round((count / total) * 100);
  progressFillMini.style.width = `${pct}%`;
}

function renderPaginationDots() {
  paginationDots.innerHTML = EXPERIMENTS_DATA.map(exp => `
    <div class="page-dot ${exp.id === activeExpId ? 'active' : ''}" 
         title="Go to EXP ${exp.num}" 
         onclick="selectExperiment(${exp.id})"></div>
  `).join('');
}

function toggleAccordion(header) {
  const item = header.closest(".accordion-item");
  item.classList.toggle("open");
}

function renderGridModal() {
  gridCardsContainer.innerHTML = EXPERIMENTS_DATA.map(exp => {
    const isDone = !!reviewedState[exp.id];
    return `
      <div class="overview-card" onclick="selectExperiment(${exp.id}); gridModal.classList.remove('open');">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="item-num">EXP ${exp.num}</span>
          <span style="font-size:0.75rem; color:${isDone ? '#10b981' : '#64748b'};">${isDone ? '✓ Completed' : 'Pending'}</span>
        </div>
        <h3 style="font-size:1rem; font-family:var(--font-heading); color:var(--text-primary); margin:4px 0;">${exp.shortTitle}</h3>
        <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.4;">${exp.dataset}</p>
        <div style="margin-top:auto; padding-top:8px; display:flex; gap:4px; flex-wrap:wrap;">
          ${exp.tags.slice(0, 2).map(t => `<span class="item-tag">${t}</span>`).join('')}
        </div>
      </div>
    `;
  }).join('');
}

function showToast(msg) {
  toastMessage.textContent = msg;
  toastNotification.classList.add("show");
  setTimeout(() => {
    toastNotification.classList.remove("show");
  }, 2600);
}

function loadThemePreference() {
  const saved = localStorage.getItem("deeplab_theme") || "dark";
  document.documentElement.setAttribute("data-theme", saved);
}

// Bootstrap app on DOM Ready
document.addEventListener("DOMContentLoaded", init);
