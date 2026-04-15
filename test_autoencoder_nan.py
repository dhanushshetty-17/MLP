"""
Test autoencoder with 10 epochs to check for NaN issues
"""
import numpy as np
from data_loader import get_fashion_mnist
from autoencoder import DenseAutoencoder

print("=" * 70)
print("AUTOENCODER NaN TEST (10 epochs)")
print("=" * 70)

# Load data
print("\nLoading data...")
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_fashion_mnist(validation_split=0.2)
print(f"  [OK] Train: {X_train.shape}")

# Use subset for faster testing
X_train_small = X_train[:10000]
X_val_small = X_val[:2000]

print("\nTraining Autoencoder (10 epochs)...")
print("-" * 70)

autoencoder = DenseAutoencoder(
    input_size=784,
    hidden_sizes=[256],
    latent_size=32,
    learning_rate=0.001,  # Fixed learning rate
    activation='relu'
)

try:
    history = autoencoder.train(
        X_train_small,
        epochs=10,
        batch_size=64,
        validation_data=X_val_small,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("[SUCCESS] No NaN detected! Training completed successfully.")
    print("=" * 70)
    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
except ValueError as e:
    print("\n" + "=" * 70)
    print(f"[FAILED] {e}")
    print("=" * 70)
