# Quick Start Guide

This document provides a quick overview of how to get started with this deep learning project.

## Prerequisites

- Python 3.8 or higher
- 2GB free disk space (for dataset download)
- 4GB RAM (recommended)

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python verify_setup.py
```

You should see:
```
✓ Python Version: 3.x.x
✓ Checking dependencies:
  ✓ NumPy - OK
  ✓ Matplotlib - OK
  ✓ Scikit-learn - OK
✓ Checking project modules:
  ✓ utils - OK
  ✓ data_loader - OK
  ✓ mlp - OK
  ✓ autoencoder - OK
  ✓ rbm - OK
ALL CHECKS PASSED!
```

## Running the Project

### Option 1: Train All Models (Recommended for First Run)
```bash
python train.py
```

This will:
- Download Fashion-MNIST dataset (∼220MB)
- Train MLP for classification (∼2-3 minutes)
- Train Dense Autoencoder (∼2-3 minutes)
- Train RBM (∼3-4 minutes)
- Generate visualizations
- Total time: ∼10-15 minutes on CPU

**Output files created in `results/` directory:**
- `training_history.png` - Training curves for all models
- `reconstructions.png` - Autoencoder reconstruction examples
- `rbm_filters.png` - Learned RBM filters

### Option 2: Detailed Evaluation
```bash
python evaluate.py
```

This will:
- Run MLP evaluation with confusion matrix
- Analyze Autoencoder reconstruction error
- Analyze RBM feature learning
- Generate detailed performance reports

**Additional output files:**
- `mlp_confusion_matrix.png` - Classification confusion matrix
- `autoencoder_analysis.png` - Reconstruction error distribution
- `rbm_analysis.png` - Feature activation analysis

## Expected Performance

After running `train.py`, you should see:

```
[2/5] Training Multilayer Perceptron (MLP)...
Epoch 5/30 - Loss: 0.2543, Acc: 92.45% - Val Loss: 0.2651, Val Acc: 91.23%
Epoch 10/30 - Loss: 0.1876, Acc: 94.12% - Val Loss: 0.2145, Val Acc: 93.45%
...
Epoch 30/30 - Loss: 0.0987, Acc: 96.78% - Val Loss: 0.1654, Val Acc: 94.23%
  ✓ MLP Test Accuracy: 88.45%

[3/5] Training Dense Autoencoder...
Epoch 5/30 - Train Loss: 0.001234, Val Loss: 0.001567
...
  ✓ Autoencoder Test MSE: 0.000891

[4/5] Training Restricted Boltzmann Machine (RBM)...
Epoch 5/30 - Free Energy: -123.45
...
  ✓ RBM Test Reconstruction Error: 0.145234
```

## Project Structure

```
d:\MLP(Leslie)\
├── utils.py                 # Core utility functions
├── data_loader.py          # Fashion-MNIST loader
├── mlp.py                  # MLP implementation
├── autoencoder.py          # Autoencoder implementation
├── rbm.py                  # RBM implementation
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
├── verify_setup.py         # Verification script
├── requirements.txt        # Python dependencies
├── README.md              # Full documentation
├── QUICKSTART.md          # This file
└── results/               # Output directory (created after first run)
    ├── training_history.png
    ├── reconstructions.png
    ├── rbm_filters.png
    ├── mlp_confusion_matrix.png
    ├── autoencoder_analysis.png
    └── rbm_analysis.png
```

## Customization

### Modify Hyperparameters

Edit `train.py` to adjust:

```python
# MLP Configuration (around line 63)
mlp = MLP(
    input_size=784,
    hidden_size=256,  # Change from 128
    output_size=10,
    hidden_activation='relu',
    learning_rate=0.01  # Change from 0.05
)

# Autoencoder Configuration (around line 88)
autoencoder = DenseAutoencoder(
    input_size=784,
    hidden_sizes=[512, 256],  # Change from [256]
    latent_size=32,
    learning_rate=0.001,  # Change as needed
    activation='relu'
)

# RBM Configuration (around line 112)
rbm = RBM(
    n_visible=784,
    n_hidden=256,  # Change from 100
    learning_rate=0.001
)

# Training parameters (update all sections)
history_mlp = mlp.train(
    X_train, y_train_onehot,
    epochs=50,  # Change from 30
    batch_size=32,  # Change from 64
)
```

### Experiment with Different Architectures

**MLP:**
```python
# Try different hidden layer sizes
hidden_size = 256  # vs 128, 64, 512
hidden_activation = 'sigmoid'  # vs 'relu'
learning_rate = 0.01  # vs 0.05, 0.1
```

**Autoencoder:**
```python
# Try different encoder/decoder depths
hidden_sizes = [512, 256, 128]  # vs [256]
latent_size = 16  # vs 32, 64
sparse = True  # Enable sparsity
```

**RBM:**
```python
# Try different hidden dimensions
n_hidden = 200  # vs 100, 50
cd_k = 5  # More Gibbs sampling steps
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named X"
**Solution:** Run `pip install -r requirements.txt` again

### Issue: "CUDA out of memory"
**Solution:** Reduce `batch_size` in train.py (e.g., 32 instead of 64)

### Issue: Training is very slow
**Solution:** 
- This is expected on CPU. Consider GPU if available.
- Reduce `epochs` for quicker testing
- Reduce `batch_size` for faster iteration

### Issue: Dataset download fails
**Solution:**
- Check internet connection
- The dataset will be cached after first download
- Manual download: https://github.com/zalandoresearch/fashion-mnist

## Next Steps

1. ✓ Run `verify_setup.py` to ensure everything works
2. ✓ Run `train.py` to train all models
3. ✓ Run `evaluate.py` to see detailed analysis
4. ✓ Experiment with hyperparameters
5. ✓ Study the code to understand implementations
6. ✓ Check visualizations in `results/` directory

## Further Learning

- Modify network architectures in `mlp.py`, `autoencoder.py`, `rbm.py`
- Implement new activation functions in `utils.py`
- Add new metrics or loss functions
- Extend to other datasets
- Implement CNN or other architectures

## References

- Deep Learning textbook: https://www.deeplearningbook.org/
- Fashion-MNIST paper: https://github.com/zalandoresearch/fashion-mnist
- NumPy documentation: https://numpy.org/doc/
- Matplotlib tutorial: https://matplotlib.org/

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review code comments in implementation files
3. Run evaluate.py to debug training
4. Experiment with simpler configurations first

---

**Happy Learning!** 🚀
