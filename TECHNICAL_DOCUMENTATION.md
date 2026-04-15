# Technical Documentation & Implementation Details

## Overview

This document provides comprehensive technical details about the implementation of three fundamental deep learning models from scratch using NumPy.

## 1. Multilayer Perceptron (MLP) - Classification

### 1.1 Architecture

```
Input Layer (784 neurons)
    ↓ [Weight Matrix: 784×128]
Hidden Layer (128 neurons, ReLU activation)
    ↓ [Weight Matrix: 128×10]
Output Layer (10 neurons, Softmax activation)
```

### 1.2 Forward Propagation

```
z¹ = Xw¹ + b¹
a¹ = ReLU(z¹) = max(0, z¹)
z² = a¹w² + b²
ŷ = Softmax(z²) = exp(z²) / Σexp(z²)
```

Where:
- X: Input features (N×784)
- w¹: Hidden layer weights (784×128)
- b¹: Hidden layer biases (128,)
- w²: Output layer weights (128×10)
- b²: Output layer biases (10,)

### 1.3 Loss Function (Cross-Entropy)

```
L = -Σ(yᵢ log(ŷᵢ)) / N

Where:
- y: One-hot encoded true labels
- ŷ: Predicted probabilities
- N: Batch size
```

### 1.4 Backpropagation

**Output Layer Gradient:**
```
∂L/∂z² = ŷ - y  (derivative of softmax cross-entropy)
dw² = (1/N) × a¹ᵀ × ∂L/∂z²
db² = (1/N) × Σ(∂L/∂z²)
```

**Hidden Layer Gradient:**
```
∂L/∂a¹ = ∂L/∂z² × w²ᵀ
∂L/∂z¹ = ∂L/∂a¹ × ReLU'(z¹)  where ReLU'(z) = (z > 0)
dw¹ = (1/N) × Xᵀ × ∂L/∂z¹
db¹ = (1/N) × Σ(∂L/∂z¹)
```

**Weight Update:**
```
w ← w - lr × ∇L(w)
b ← b - lr × ∇L(b)
```

### 1.5 Activation Functions

**ReLU (Rectified Linear Unit):**
```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0, else 0

Advantages:
- Faster convergence
- Less vanishing gradient
- Sparse activation
```

**Softmax:**
```
Softmax(zᵢ) = exp(zᵢ) / Σexp(zⱼ)

Properties:
- Produces probability distribution
- Sum = 1
- Suitable for multi-class classification
```

### 1.6 Weight Initialization

**Xavier Initialization:**
```
W ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))

Purpose:
- Prevents vanishing/exploding gradients
- Maintains variance across layers
- Facilitates faster convergence
```

### 1.7 Training Algorithm (SGD)

```
for epoch in epochs:
    for batch in training_data:
        # Forward pass
        y_pred = forward(X_batch)
        
        # Compute loss
        loss = cross_entropy(y_true, y_pred)
        
        # Backward pass
        gradients = backward(loss)
        
        # Update weights
        update_weights(gradients, learning_rate)
```

### 1.8 Implementation Details

**File:** `mlp.py`

Key methods:
- `forward(X)`: Computes forward propagation
- `backward(y_true)`: Computes gradients and updates weights
- `train()`: Full training loop with validation monitoring
- `predict()`: Inference on new data

**Hyperparameters:**
```python
learning_rate = 0.05      # SGD learning rate
hidden_size = 128         # Hidden layer neurons
activation = 'relu'       # Hidden layer activation
batch_size = 64          # Mini-batch size
epochs = 30              # Training epochs
```

---

## 2. Dense Autoencoder - Reconstruction

### 2.1 Architecture

**Encoder:**
```
Input (784) 
    ↓ Dense + ReLU (256)
    ↓ Dense + Sigmoid (32)
Latent Representation (32)
```

**Decoder:**
```
Latent (32)
    ↓ Dense + ReLU (256)
    ↓ Dense + Sigmoid (784)
Output (784)
```

### 2.2 How Autoencoders Work

```
Original Image X
    ↓
ENCODER: Compress information (784 → 32)
    ↓
Latent Code z: 32-dimensional vector
    ↓
DECODER: Reconstruct from compressed code (32 → 784)
    ↓
Reconstructed Image X̂
```

### 2.3 Forward Pass

```
# Encoder Forward
z¹ = Xw₁ᵉ + b₁ᵉ
a¹ = ReLU(z¹)
z_latent = a¹w₂ᵉ + b₂ᵉ
z = Sigmoid(z_latent)  # Latent code

# Decoder Forward
z¹ = zw₁ᵈ + b₁ᵈ
a¹ = ReLU(z¹)
z_out = a¹w₂ᵈ + b₂ᵈ
X̂ = Sigmoid(z_out)  # Reconstructed output
```

### 2.4 Loss Function (MSE)

```
L = (1/N) × Σ||X - X̂||²
  = (1/N) × Σ(X - X̂)²

Interpretation:
- Measures pixel-wise reconstruction error
- Penalizes all deviations equally
- Suitable for continuous data
```

### 2.5 Backpropagation Through Autoencoder

**Output Layer Gradient:**
```
∂L/∂X̂ = 2(X̂ - X) / N
∂L/∂z_out = ∂L/∂X̂ × Sigmoid'(z_out)
```

**Decoder Backward Pass:**
```
For layer i (from output to input):
    dWᵢ = aᵢ₋₁ᵀ × ∂L/∂zᵢ / N
    dbᵢ = Σ(∂L/∂zᵢ) / N
    ∂L/∂aᵢ₋₁ = ∂L/∂zᵢ × Wᵢᵀ
    ∂L/∂zᵢ₋₁ = ∂L/∂aᵢ₋₁ × Activation'(zᵢ₋₁)
```

**Encoder Backward Pass:**
```
Similar gradient computation through encoder layers
With propagation from decoder output back to encoder input
```

### 2.6 Sparse Autoencoder (Optional)

**Sparsity Loss:**
```
L_total = L_reconstruction + λ × L_sparsity

L_sparsity = Σ KL(ρ || ρ̂)
where:
- ρ: Target sparsity (e.g., 0.05)
- ρ̂: Actual average activation
- KL: Kullback-Leibler divergence
```

**Effect:**
- Forces most hidden units to be inactive
- Results in feature detection
- Better generalization
- Sparse representations

### 2.7 Undercomplete vs Overcomplete

```
Undercomplete (latent_size < input_size):
- Bottleneck forces compression
- Learns efficient representations
- Good for dimensionality reduction

Overcomplete (latent_size > input_size):
- Risk of learning identity
- Requires regularization
- Can learn all details
```

### 2.8 Applications

**Dimensionality Reduction:**
```
Original: 784 dimensions
Latent: 32 dimensions (95% compression)
Use z for downstream tasks
```

**Outlier Detection:**
```
Normal samples: Low reconstruction error
Anomalies: High reconstruction error
Threshold = mean + 2×std(error)
```

**Feature Learning:**
```
Latent representation z captures essential features
Can be visualized with t-SNE
Useful for clustering/classification
```

### 2.9 Implementation Details

**File:** `autoencoder.py`

Key methods:
- `encode(X)`: Forward through encoder only
- `decode(latent)`: Forward through decoder only
- `forward(X)`: Complete forward pass
- `backward()`: Backpropagation through both encoder and decoder
- `get_reconstruction_error()`: Compute error for each sample

---

## 3. Restricted Boltzmann Machine (RBM)

### 3.1 Architecture

```
Visible Units (784)
    ↕ Connections (No hidden-hidden connections)
Hidden Units (100)

Undirected graph with:
- Visible to Hidden: Bidirectional
- No Hidden to Hidden: Independent given visible
- No Visible to Visible: Independent given hidden
```

### 3.2 Energy-Based Model

```
Energy function:
E(v,h) = -vᵀWh - vᵀbᵥ - hᵀbₕ

Probability distribution:
P(v,h) = exp(-E(v,h)) / Z(θ)

Marginal probabilities:
P(v) = Σₕ P(v,h)
P(h) = Σᵥ P(v,h)
```

Where:
- v: Visible units (binary)
- h: Hidden units (binary)
- W: Connection weights (784×100)
- bᵥ: Visible biases
- bₕ: Hidden biases

### 3.3 Conditional Probabilities

**Given Visible (Bottom-Up):**
```
P(h = 1 | v) = σ(Wᵀv + bₕ)
where σ is sigmoid activation

This is efficient to compute:
- Each hidden unit can be computed independently
- No hidden-hidden connections
```

**Given Hidden (Top-Down):**
```
P(v = 1 | h) = σ(Wh + bᵥ)

Similarly efficient:
- Each visible unit is independent given h
- Used for reconstruction/generation
```

### 3.4 Contrastive Divergence (CD-1) Learning

**Problem:** Computing gradients requires:
1. Sample from P(v,h) - Computationally expensive
2. Numerical approximation - Intractable

**Solution: Contrastive Divergence**

```
Algorithm CD-1:

1. Positive Phase (Data-dependent):
   - Start with training sample v₀
   - Sample h₀ ~ P(h | v₀)  [1 step]
   - Compute outer product: v₀h₀ᵀ

2. Negative Phase (Model-dependent):
   - Sample v₁ ~ P(v | h₀)  [Gibbs sampling]
   - Sample h₁ ~ P(h | v₁)
   - Compute outer product: v₁h₁ᵀ

3. Weight Update:
   ΔW = lr × [(v₀h₀ᵀ) - (v₁h₁ᵀ)] / N_batch
   
4. Bias Updates:
   Δbᵥ = lr × (v₀ - v₁) / N_batch
   Δbₕ = lr × (h₀ - h₁) / N_batch
```

### 3.5 Gibbs Sampling

```
Alternating sampling between v and h:

1. Start with v₀
2. h₀ ~ P(h | v₀)
3. v₁ ~ P(v | h₀)
4. h₁ ~ P(h | v₁)
5. v₂ ~ P(v | h₁)
...

CD-k means performing k steps of Gibbs sampling:
- CD-1: 1 step (fast, good in practice)
- CD-5: 5 steps (more accurate, slower)
- CD-25: 25 steps (very accurate, very slow)
```

### 3.6 Free Energy (Convergence Monitoring)

```
Free Energy:
F(v) = -vᵀbᵥ - Σⱼ log(1 + exp(Wⱼᵀv + bₕⱼ))

Properties:
- More negative = better lower probability mass
- Should decrease during training
- Used to monitor learning progress
```

### 3.7 Feature Extraction

**Transformation:**
```
Input: v (784-dimensional visible vector)
Hidden activations: h = σ(Wᵀv + bₕ)  (100-dim)
Output: 100-dimensional feature representation
```

**Interpretation:**
```
Each hidden unit learns a detector:
- Edge patterns
- Texture features
- Combinations of pixels
```

### 3.8 Generation (Reconstruction)

```
Given input v:
1. Sample h ~ P(h | v)
2. Sample v' ~ P(v | h)
3. Sample h' ~ P(h | v')
...

Results:
- v → v': Reconstructed from learned distribution
- Artifacts smooth out noise
- Useful for denoising
```

### 3.9 Implementation Details

**File:** `rbm.py`

Key methods:
- `sample_h_given_v(v)`: Compute P(h|v) and sample
- `sample_v_given_h(h)`: Compute P(v|h) and sample
- `contrastive_divergence(v, k)`: Perform CD-k algorithm
- `train()`: Main training loop
- `transform(X)`: Extract hidden features
- `reconstruct(X)`: Generate from learned distribution

**Key Hyperparameters:**
```python
n_hidden = 100        # Hidden units
learning_rate = 0.01  # CD learning rate
cd_k = 1             # Contrastive divergence steps
batch_size = 64      # Mini-batch size
epochs = 30          # Training epochs
```

---

## 4. Training Dynamics & Optimization

### 4.1 Mini-Batch SGD

```
Algorithm:
for epoch in epochs:
    shuffle training data
    for batch in minibatches:
        # Compute gradient on batch
        g = ∇L(θ; batch)
        
        # Update parameters
        θ ← θ - lr × g
```

**Advantages:**
- Faster than full-batch
- Better generalization than SGD
- Efficient GPU/parallel computation
- Reduces variance

### 4.2 Learning Rate Decay

```
lr(t) = lr₀ / (1 + decay_rate × t)

Effect:
- Start with larger steps (fast convergence)
- Gradually decrease step size
- Fine-tune near optimum
- Prevents oscillation
```

### 4.3 Convergence Monitoring

**Validation Strategy:**
```
Training Loop:
1. Train on training set
2. Evaluate on validation set
3. Monitor loss and accuracy
4. Early stopping if validation loss increases

Prevents:
- Overfitting
- Excessive training time
```

---

## 5. Common Issues & Solutions

### 5.1 Vanishing Gradients

**Problem:**
```
∂L/∂W₁ = ∂L/∂a⁵ × ∂a⁵/∂z⁵ × ... × ∂z¹/∂W₁

If each derivative < 0.1:
Product becomes exponentially small
```

**Solutions Implemented:**
```
1. ReLU activation: ∂ReLU/∂z = 1 (no scaling)
2. Xavier initialization: Maintains variance
3. Careful layer scoping: Not too deep
4. Skip connections (future work)
```

### 5.2 Numerical Stability

**Problem:**
```
softmax(z) = exp(z) / Σexp(z)
If z contains large values → overflow
```

**Solution Implemented:**
```python
# Numerically stable softmax
e_x = np.exp(x - np.max(x, keepdims=True))
softmax = e_x / np.sum(e_x, keepdims=True)

# Clipped sigmoid
sigmoid(x) = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

### 5.3 Weight Initialization

**Problem:**
```
Random small init: Slow gradients, slow learning
Random large init: Exploding gradients, divergence
```

**Solution:**
```python
# Xavier initialization
limit = np.sqrt(6 / (n_in + n_out))
W = np.random.uniform(-limit, limit, shape)

Ensures: Var(outputs) ≈ Var(inputs)
```

---

## 6. Performance Metrics

### 6.1 Classification (MLP)

**Accuracy:**
```
Acc = (TP + TN) / (TP + TN + FP + FN)
Suitable for balanced datasets
```

**Precision & Recall:**
```
Precision = TP / (TP + FP)  - Avoid false positives
Recall = TP / (TP + FN)     - Avoid false negatives
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
Harmonic mean of precision and recall
```

### 6.2 Reconstruction (Autoencoder)

**MSE:**
```
MSE = (1/N) × Σ(X - X̂)²
Pixel-level error: Good for image quality
```

**MAE:**
```
MAE = (1/N) × Σ|X - X̂|
More robust to outliers
```

**SSIM (Structural Similarity):**
```
Considers structure preservation
More aligned with human perception
(Could be added to evaluation)
```

### 6.3 Generative Models (RBM)

**Free Energy:**
```
F(v) = -vᵀbᵥ - Σ log(1 + exp(Wⱼᵀv + bₕⱼ))
Lower is better (more likely data)
```

**Log-Likelihood:**
```
Approximated as: -F(v) - log(Z)
(Partition function Z is intractable)
```

---

## 7. Code Examples

### 7.1 Using the MLP

```python
from mlp import MLP
from utils import one_hot_encode
from data_loader import get_fashion_mnist

# Load data
(X_train, y_train), _, (X_test, y_test) = get_fashion_mnist()

# Create model
mlp = MLP(
    input_size=784,
    hidden_size=256,
    output_size=10,
    learning_rate=0.05
)

# Prepare labels
y_train_onehot = one_hot_encode(y_train, 10)

# Train
history = mlp.train(X_train, y_train_onehot, epochs=30)

# Predict
predictions = mlp.predict(X_test)
probabilities = mlp.predict_proba(X_test)
```

### 7.2 Using the Autoencoder

```python
from autoencoder import DenseAutoencoder
from data_loader import get_fashion_mnist

# Load data
(X_train, _), _, (X_test, _) = get_fashion_mnist()

# Create model
ae = DenseAutoencoder(
    input_size=784,
    hidden_sizes=[256, 128],
    latent_size=32,
    sparse=True
)

# Train
history = ae.train(X_train, epochs=30)

# Use for feature extraction
features = ae.encode_data(X_test)  # (N, 32)

# Use for reconstruction
reconstructed = ae.reconstruct(X_test)

# Use for outlier detection
errors = ae.get_reconstruction_error(X_test)
threshold = np.mean(errors) + 2 * np.std(errors)
outliers = errors > threshold
```

### 7.3 Using the RBM

```python
from rbm import RBM
from data_loader import get_fashion_mnist

# Load data
(X_train, _), _, (X_test, _) = get_fashion_mnist()

# Create model
rbm = RBM(n_visible=784, n_hidden=100)

# Train
history = rbm.train(X_train, epochs=30)

# Extract features
features = rbm.transform(X_test)  # (N, 100)

# Generate reconstructions
reconstructed = rbm.reconstruct(X_test, steps=5)

# Get reconstruction error
errors = rbm.get_reconstruction_error(X_test)
```

---

## 8. Extensions & Future Work

### 8.1 Potential Improvements

```
1. Convolutional Layers
   - Better for image data
   - Reduced parameters
   - Spatial consistency

2. Advanced Optimizers
   - Adam: Adaptive learning rate
   - RMSprop:  Root mean square propagation
   - Momentum: Accelerated convergence

3. Regularization
   - Dropout: Prevent co-adaptation
   - Batch Normalization: Stabilize training
   - L1/L2: Weight penalties

4. Advanced Architectures
   - Deep Belief Networks (stack RBMs)
   - Variational Autoencoders (VAE)
   - Generative Adversarial Networks (GAN)
```

### 8.2 Research Directions

```
1. Transfer Learning
   - Pre-train on large dataset
   - Fine-tune on task-specific data

2. Meta-Learning
   - Learn to learn
   - Few-shot learning

3. Self-Supervised Learning
   - Learn without labels
   - Contrastive learning

4. Interpretability
   - Attention mechanisms
   - Feature importance
```

---

## Summary

This implementation provides:

✓ **Complete from-scratch implementations** without high-level frameworks
✓ **Mathematical rigor** with detailed gradient derivations
✓ **Educational value** for understanding deep learning
✓ **Practical applicability** for real datasets
✓ **Extensibility** for further research

All models are trained on Fashion-MNIST and achieve competitive performance while maintaining code clarity and educational value.

