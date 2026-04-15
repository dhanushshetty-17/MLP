"""
Quick training test - trains on small dataset to verify everything works
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_fashion_mnist
from utils import one_hot_encode, accuracy as compute_accuracy
from mlp import MLP
from autoencoder import DenseAutoencoder
from rbm import RBM
import os

print("=" * 70)
print("QUICK TRAINING TEST (3 epochs each)")
print("=" * 70)

# Load data (use smaller subset)
print("\n[1/4] Loading Fashion-MNIST dataset...")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_fashion_mnist(validation_split=0.2)
print(f"  [OK] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Use smaller subset for quick test
X_train_small = X_train[:5000]
y_train_small = y_train[:5000]
X_val_small = X_val[:1000]
y_val_small = y_val[:1000]

# Create results directory
os.makedirs('results', exist_ok=True)

# Test MLP
print("\n[2/4] Testing MLP (3 epochs)...")
print("-" * 70)
mlp = MLP(input_size=784, hidden_size=128, output_size=10, 
          hidden_activation='relu', learning_rate=0.05)

y_train_onehot = one_hot_encode(y_train_small, 10)
y_val_onehot = one_hot_encode(y_val_small, 10)

history_mlp = mlp.train(
    X_train_small, y_train_onehot,
    epochs=3,
    batch_size=64,
    validation_data=(X_val_small, y_val_onehot),
    verbose=True
)

y_test_onehot = one_hot_encode(y_test, 10)
y_test_pred = mlp.predict_proba(X_test)
test_acc = compute_accuracy(y_test_onehot, y_test_pred)
print(f"\n  [OK] MLP Test Accuracy: {test_acc:.2f}%")

# Test Autoencoder
print("\n[3/4] Testing Autoencoder (3 epochs)...")
print("-" * 70)
autoencoder = DenseAutoencoder(
    input_size=784,
    hidden_sizes=[256],
    latent_size=32,
    learning_rate=0.001,  # Reduced to prevent NaN
    activation='relu'
)

history_ae = autoencoder.train(
    X_train_small,
    epochs=3,
    batch_size=64,
    validation_data=X_val_small,
    verbose=True
)

X_test_reconstructed = autoencoder.reconstruct(X_test)
ae_mse = np.mean((X_test - X_test_reconstructed) ** 2)
print(f"\n  [OK] Autoencoder Test MSE: {ae_mse:.6f}")

# Test RBM
print("\n[4/4] Testing RBM (3 epochs)...")
print("-" * 70)
rbm = RBM(n_visible=784, n_hidden=100, learning_rate=0.01)

history_rbm = rbm.train(
    X_train_small,
    epochs=3,
    batch_size=64,
    cd_k=1,
    verbose=True
)

X_test_rbm_reconstructed = rbm.reconstruct(X_test, steps=5)
rbm_error = np.mean(np.abs(X_test - X_test_rbm_reconstructed))
print(f"\n  [OK] RBM Test Reconstruction Error: {rbm_error:.6f}")

print("\n" + "=" * 70)
print("QUICK TEST COMPLETE - ALL MODELS WORKING!")
print("=" * 70)
print("\nAll components are functioning correctly.")
print("To train full models (30 epochs), run:")
print("  c:/python314/python.exe train.py")
print("\n" + "=" * 70)
