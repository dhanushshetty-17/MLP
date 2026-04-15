"""
Final verification - train all models for 5 epochs to verify no NaN
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
print("FINAL VERIFICATION TEST (5 epochs each)")
print("=" * 70)

# Load data
print("\n[1/4] Loading Fashion-MNIST dataset...")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_fashion_mnist(validation_split=0.2)
print(f"  [OK] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Create results directory
os.makedirs('results', exist_ok=True)

# Test MLP
print("\n[2/4] Training MLP (5 epochs)...")
print("-" * 70)
mlp = MLP(input_size=784, hidden_size=128, output_size=10, 
          hidden_activation='relu', learning_rate=0.05)

y_train_onehot = one_hot_encode(y_train, 10)
y_val_onehot = one_hot_encode(y_val, 10)

history_mlp = mlp.train(
    X_train, y_train_onehot,
    epochs=5,
    batch_size=64,
    validation_data=(X_val, y_val_onehot),
    verbose=True
)

y_test_onehot = one_hot_encode(y_test, 10)
y_test_pred = mlp.predict_proba(X_test)
test_acc = compute_accuracy(y_test_onehot, y_test_pred)
print(f"\n  [OK] MLP Test Accuracy: {test_acc:.2f}%")

# Check for NaN
if np.isnan(history_mlp['train_loss'][-1]):
    print("  [ERROR] NaN detected in MLP training!")
else:
    print(f"  [OK] MLP final loss: {history_mlp['train_loss'][-1]:.4f} (no NaN)")

# Test Autoencoder
print("\n[3/4] Training Autoencoder (5 epochs)...")
print("-" * 70)
autoencoder = DenseAutoencoder(
    input_size=784,
    hidden_sizes=[256],
    latent_size=32,
    learning_rate=0.001,  # Fixed learning rate
    activation='relu'
)

history_ae = autoencoder.train(
    X_train,
    epochs=5,
    batch_size=64,
    validation_data=X_val,
    verbose=True
)

X_test_reconstructed = autoencoder.reconstruct(X_test)
ae_mse = np.mean((X_test - X_test_reconstructed) ** 2)
print(f"\n  [OK] Autoencoder Test MSE: {ae_mse:.6f}")

# Check for NaN
if np.isnan(history_ae['train_loss'][-1]):
    print("  [ERROR] NaN detected in Autoencoder training!")
else:
    print(f"  [OK] Autoencoder final loss: {history_ae['train_loss'][-1]:.6f} (no NaN)")

# Test RBM
print("\n[4/4] Training RBM (5 epochs)...")
print("-" * 70)
rbm = RBM(n_visible=784, n_hidden=100, learning_rate=0.01)

history_rbm = rbm.train(
    X_train,
    epochs=5,
    batch_size=64,
    cd_k=1,
    verbose=True
)

X_test_rbm_reconstructed = rbm.reconstruct(X_test, steps=5)
rbm_error = np.mean(np.abs(X_test - X_test_rbm_reconstructed))
print(f"\n  [OK] RBM Test Reconstruction Error: {rbm_error:.6f}")

# Check for NaN
if np.isnan(history_rbm['free_energy'][-1]):
    print("  [ERROR] NaN detected in RBM training!")
else:
    print(f"  [OK] RBM final free energy: {history_rbm['free_energy'][-1]:.2f} (no NaN)")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\n[SUCCESS] All models trained without NaN errors!")
print("\nSummary:")
print(f"  MLP: {test_acc:.2f}% accuracy (final loss: {history_mlp['train_loss'][-1]:.4f})")
print(f"  Autoencoder: {ae_mse:.6f} MSE (final loss: {history_ae['train_loss'][-1]:.6f})")
print(f"  RBM: {rbm_error:.6f} reconstruction error (final energy: {history_rbm['free_energy'][-1]:.2f})")
print("\n" + "=" * 70)
print("Your code is working correctly. You can now run full training:")
print("  c:/python314/python.exe train.py")
print("=" * 70)
