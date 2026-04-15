"""Quick Autoencoder-only test to verify NaN fix"""
import numpy as np
from data_loader import get_fashion_mnist
from autoencoder import DenseAutoencoder

print("Testing Autoencoder (3 epochs)...\n")

# Load small subset
(X_train, y_train), (X_val, y_val), (X_test, y_test) = get_fashion_mnist(validation_split=0.2)

# Use 10% of data for speed
np.random.seed(42)
train_idx = np.random.choice(len(X_train), size=4800, replace=False)
val_idx = np.random.choice(len(X_val), size=1200, replace=False)

X_train_small = X_train[train_idx]
X_val_small = X_val[val_idx]

# Train
autoencoder = DenseAutoencoder(
    input_size=784,
    hidden_sizes=[256],
    latent_size=32,
    learning_rate=0.001,
    activation='relu'
)

history = autoencoder.train(
    X_train_small,
    epochs=3,
    batch_size=64,
    validation_data=X_val_small,
    verbose=True
)

# Check results
final_loss = history['train_loss'][-1]
final_val_loss = history['val_loss'][-1]

print(f"\nFinal Train Loss: {final_loss:.6f}")
print(f"Final Val Loss: {final_val_loss:.6f}")

if np.isnan(final_loss) or np.isnan(final_val_loss):
    print("\n[FAILED] NaN detected!")
else:
    print("\n[SUCCESS] No NaN - Autoencoder is working correctly!")
    
    # Test reconstruction
    X_test_small = X_test[:100]
    X_reconstructed = autoencoder.reconstruct(X_test_small)
    mse = np.mean((X_test_small - X_reconstructed) ** 2)
    print(f"Test MSE: {mse:.6f}")
