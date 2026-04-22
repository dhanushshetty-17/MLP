# Deep Learning Assignment - From Scratch Implementation

A comprehensive implementation of fundamental deep learning models using Python and NumPy, including:
- **Multilayer Perceptron (MLP)** for classification
- **Dense Autoencoder** for reconstruction and feature learning  
- **Restricted Boltzmann Machine (RBM)** for unsupervised feature learning

## Dataset

**Fashion-MNIST**: A dataset of 70,000 28×28 grayscale images of 10 fashion classes
- Training: 50,000 samples
- Validation: 10,000 samples
- Testing: 10,000 samples

## Project Structure

```
├── utils.py              # Utility functions (activations, loss functions, initializers)
├── data_loader.py        # Fashion-MNIST data loading and preprocessing
├── mlp.py                # MLP classifier implementation
├── autoencoder.py        # Dense Autoencoder implementation
├── rbm.py                # Restricted Boltzmann Machine implementation
├── train.py              # Main training script for all models
├── evaluate.py           # Evaluation and analysis script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# OR
source venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training All Models

```bash
python train.py
```

This will:
1. Load Fashion-MNIST dataset
2. Train MLP classifier (30 epochs)
3. Train Dense Autoencoder (30 epochs)
4. Train RBM (30 epochs)
5. Generate visualizations (training curves, reconstructions, filters)

### Detailed Evaluation

```bash
python evaluate.py
```

This provides:
- Detailed classification metrics for MLP
- Confusion matrix and per-class accuracy
- Autoencoder reconstruction analysis and outlier detection
- RBM feature analysis and filter visualization
- Comprehensive performance report

## Model Architectures

### MLP (Classification)
```
Input (784)
    ↓
Dense Layer + ReLU (128)
    ↓
Dense Layer + Softmax (10)
    ↓
Output (10 classes)
```

**Training Details:**
- Loss: Cross-Entropy
- Optimizer: SGD
- Learning Rate: 0.05
- Batch Size: 64
- Epochs: 30

**Expected Performance:**
- Test Accuracy: ~88%

### Autoencoder (Reconstruction)
```
Encoder:
Input (784) → Dense + ReLU (256) → Dense + Sigmoid (32)

Decoder:
Latent (32) → Dense + ReLU (256) → Dense + Sigmoid (784)
```

**Training Details:**
- Loss: Mean Squared Error
- Optimizer: SGD
- Learning Rate: 0.01
- Batch Size: 64
- Epochs: 30

**Features:**
- Undercomplete autoencoder for dimensionality reduction
- Optional sparse variant with L1 regularization
- Outlier detection using reconstruction error

### RBM (Unsupervised Learning)
```
Visible Units: 784
    ↕ (Bidirectional Connections)
Hidden Units: 100
```

**Training Details:**
- Learning Algorithm: Contrastive Divergence (CD-1)
- Learning Rate: 0.01
- Batch Size: 64
- Epochs: 30

**Features:**
- Learns probabilistic representations
- Extracts meaningful visual features
- Can reconstruct input from hidden features

## Key Implementation Details

### MLP
- Forward propagation through two layers
- Backpropagation with gradient computation
- Xavier weight initialization
- ReLU activation for faster convergence
- Softmax output for multi-class classification

### Autoencoder
- Symmetric encoder-decoder architecture
- Flexible number of hidden layers
- MSE loss for continuous reconstruction
- Optional sparsity regularization
- Reconstruction error for outlier detection

### RBM
- Contrastive Divergence learning algorithm
- Gibbs sampling for inference
- Energy-based probabilistic model
- Feature extraction and dimensionality reduction
- Visualization of learned filters

## Performance Metrics

### MLP
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Per-class performance breakdown

### Autoencoder
- Mean Squared Error (MSE)
- Reconstruction error distribution
- Outlier detection rate
- Latent space visualization

### RBM
- Free Energy (convergence monitoring)
- Reconstruction Error (MAE)
- Feature activation statistics
- Filter visualization

## Hyperparameter Tuning

Key hyperparameters to experiment with:

**MLP:**
- `hidden_size`: Number of hidden units (tested: 64, 128, 256)
- `learning_rate`: SGD learning rate (tested: 0.01, 0.05, 0.1)
- `hidden_activation`: 'relu' or 'sigmoid'

**Autoencoder:**
- `hidden_sizes`: List of encoder hidden layer sizes
- `latent_size`: Dimensionality of latent representation
- `learning_rate`: Optimizer learning rate
- `sparse`: Enable/disable sparsity regularization

**RBM:**
- `n_hidden`: Number of hidden units (tested: 50, 100, 200)
- `learning_rate`: Learning rate for weight updates
- `cd_k`: Number of Gibbs sampling steps

## Results and Outputs

Running the training script generates:

1. **Training History** (`results/training_history.png`)
   - MLP loss and accuracy curves
   - Autoencoder MSE curves
   - RBM free energy curve

2. **Autoencoder Reconstructions** (`results/reconstructions.png`)
   - Side-by-side comparison of original and reconstructed images

3. **RBM Filters** (`results/rbm_filters.png`)
   - 16 learned hidden unit filters
   - Shows edge-like patterns captured by RBM

Running the evaluation script generates:

4. **MLP Confusion Matrix** (`results/mlp_confusion_matrix.png`)
   - Per-class classification performance

5. **Autoencoder Analysis** (`results/autoencoder_analysis.png`)
   - Reconstruction error distribution
   - Latent space visualization

6. **RBM Analysis** (`results/rbm_analysis.png`)
   - Hidden unit activation distribution
   - Mean activation per unit

## Mathematical Foundations

### Forward Pass (MLP)
```
z¹ = Xw¹ + b¹
a¹ = ReLU(z¹)
z² = a¹w² + b²
a² = Softmax(z²)
```

### Backpropagation
```
∂L/∂w² = (1/m) × a¹ᵀ(a² - y)
∂L/∂w¹ = (1/m) × Xᵀ(∂L/∂z¹)
```

### Autoencoder Loss
```
L = MSE(X, X̂) = (1/m) × Σ||X - X̂||²
```

### RBM Energy Function
```
E(v,h) = -vᵀWh - vᵀbᵥ - hᵀbₕ
P(v,h) = exp(-E(v,h)) / Z
```

## Technology Stack

- **Language**: Python 3.8+
- **Core Library**: NumPy (matrix operations)
- **Visualization**: Matplotlib
- **Metrics**: Scikit-learn (confusion matrix, metrics)
- **Hardware**: CPU-based training

## Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Vanishing Gradients | Use ReLU activation and proper weight initialization |
| Slow Convergence | Learning rate tuning and batch normalization |
| Backprop Debugging | Numerical gradient checking and careful implementation |
| Memory Efficiency | Mini-batch processing and efficient NumPy operations |
| Hyperparameter Selection | Systematic grid search and validation curves |

## References

[1] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
[2] Zalando Research, *Fashion-MNIST Dataset*, https://github.com/zalandoresearch/fashion-mnist
[3] C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
[4] G. Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*, Neural Computation, 2002.
[5] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, *Gradient-Based Learning Applied to Document Recognition*, Proceedings of IEEE, 1998.

## Limitations and Future Work

### Current Limitations
- CPU-only training (slower than GPU)
- Basic network architectures
- Limited to small datasets
- No advanced optimizers (Adam, RMSProp)

### Future Enhancements
- GPU acceleration using CuPy
- Convolutional Autoencoder
- Variational Autoencoder (VAE)
- Deep Belief Networks
- Transfer learning
- Batch normalization
- Dropout regularization
- More sophisticated optimizers

## Author
**Student Name**: Dhanush Shetty
                  Dusica S
                  D Chitra

## License
Educational project - MIT License

## Acknowledgments
- Fashion-MNIST dataset by Zalando Research
- NumPy documentation and tutorials
- Deep Learning textbook by Goodfellow et al.
