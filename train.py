"""
Main training script for all models
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_fashion_mnist
from utils import one_hot_encode, accuracy as compute_accuracy
from mlp import MLP
from autoencoder import DenseAutoencoder
from rbm import RBM


def plot_training_history(histories: dict, save_path: str = 'results/training_history.png'):
    """Plot training histories for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Training Progress', fontsize=16, fontweight='bold')
    
    # MLP Loss
    ax = axes[0, 0]
    ax.plot(histories['mlp']['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in histories['mlp']:
        ax.plot(histories['mlp']['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('MLP - Cross Entropy Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MLP Accuracy
    ax = axes[0, 1]
    ax.plot(histories['mlp']['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in histories['mlp']:
        ax.plot(histories['mlp']['val_acc'], label='Val Accuracy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('MLP - Classification Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Autoencoder Loss
    ax = axes[1, 0]
    ax.plot(histories['autoencoder']['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in histories['autoencoder']:
        ax.plot(histories['autoencoder']['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Autoencoder - Reconstruction Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RBM Free Energy
    ax = axes[1, 1]
    ax.plot(histories['rbm']['free_energy'], label='Free Energy', linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Free Energy')
    ax.set_title('RBM - Free Energy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Training history saved to {save_path}")
    plt.close()


def visualize_reconstructions(autoencoder, X_test: np.ndarray,
                              save_path: str = 'results/reconstructions.png'):
    """Visualize autoencoder reconstructions."""
    n_samples = 8
    X_reconstructed = autoencoder.reconstruct(X_test[:n_samples])
    
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
    fig.suptitle('Autoencoder Reconstructions (Top: Original, Bottom: Reconstructed)',
                 fontsize=14, fontweight='bold')
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original {i+1}', fontsize=9)
        
        # Reconstructed
        axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Reconstructed {i+1}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Reconstructions saved to {save_path}")
    plt.close()


def visualize_rbm_filters(rbm, save_path: str = 'results/rbm_filters.png'):
    """Visualize RBM hidden unit filters."""
    W = rbm.get_weights_visualization()
    n_filters = min(16, W.shape[1])
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('RBM Learned Filters', fontsize=14, fontweight='bold')
    
    for i in range(n_filters):
        ax = axes[i // 4, i % 4]
        filter_img = W[:, i].reshape(28, 28)
        # Normalize for visualization
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        ax.imshow(filter_img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i+1}', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] RBM filters saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("DEEP LEARNING FROM SCRATCH - Fashion-MNIST")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading Fashion-MNIST dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_fashion_mnist(validation_split=0.2)
    print(f"  [OK] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    histories = {}
    
    # Train MLP
    print("\n[2/5] Training Multilayer Perceptron (MLP)...")
    print("-" * 70)
    mlp = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        hidden_activation='relu',
        learning_rate=0.05
    )
    
    y_train_onehot = one_hot_encode(y_train, 10)
    y_val_onehot = one_hot_encode(y_val, 10)
    
    history_mlp = mlp.train(
        X_train, y_train_onehot,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val_onehot),
        verbose=True
    )
    histories['mlp'] = history_mlp
    
    # Evaluate MLP
    y_test_onehot = one_hot_encode(y_test, 10)
    y_test_pred = mlp.predict_proba(X_test)
    test_acc = compute_accuracy(y_test_onehot, y_test_pred)
    print(f"\n  [OK] MLP Test Accuracy: {test_acc:.2f}%")
    
    # Train Autoencoder
    print("\n[3/5] Training Dense Autoencoder...")
    print("-" * 70)
    autoencoder = DenseAutoencoder(
        input_size=784,
        hidden_sizes=[256],
        latent_size=32,
        learning_rate=0.001,  # Reduced from 0.01 to prevent NaN
        activation='relu',
        sparse=False
    )
    
    history_ae = autoencoder.train(
        X_train,
        epochs=30,
        batch_size=64,
        validation_data=X_val,
        verbose=True
    )
    histories['autoencoder'] = history_ae
    
    # Evaluate Autoencoder
    X_test_reconstructed = autoencoder.reconstruct(X_test)
    ae_mse = np.mean((X_test - X_test_reconstructed) ** 2)
    print(f"\n  [OK] Autoencoder Test MSE: {ae_mse:.6f}")
    
    # Train RBM
    print("\n[4/5] Training Restricted Boltzmann Machine (RBM)...")
    print("-" * 70)
    rbm = RBM(
        n_visible=784,
        n_hidden=100,
        learning_rate=0.01
    )
    
    history_rbm = rbm.train(
        X_train,
        epochs=30,
        batch_size=64,
        cd_k=1,
        verbose=True
    )
    histories['rbm'] = history_rbm
    
    # Evaluate RBM
    X_test_rbm_reconstructed = rbm.reconstruct(X_test, steps=5)
    rbm_error = np.mean(np.abs(X_test - X_test_rbm_reconstructed))
    print(f"\n  [OK] RBM Test Reconstruction Error: {rbm_error:.6f}")
    
    # Visualizations
    print("\n[5/5] Generating visualizations...")
    print("-" * 70)
    plot_training_history(histories, save_path='results/training_history.png')
    visualize_reconstructions(autoencoder, X_test, save_path='results/reconstructions.png')
    visualize_rbm_filters(rbm, save_path='results/rbm_filters.png')
    
    # Summary Report
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nMLP Classification:")
    print(f"  - Final Train Accuracy: {history_mlp['train_acc'][-1]:.2f}%")
    print(f"  - Final Val Accuracy: {history_mlp['val_acc'][-1]:.2f}%")
    print(f"  - Test Accuracy: {test_acc:.2f}%")
    
    print(f"\nAutoencoder Reconstruction:")
    print(f"  - Final Train Loss: {history_ae['train_loss'][-1]:.6f}")
    print(f"  - Final Val Loss: {history_ae['val_loss'][-1]:.6f}")
    print(f"  - Test MSE: {ae_mse:.6f}")
    
    print(f"\nRBM Feature Learning:")
    print(f"  - Final Free Energy: {history_rbm['free_energy'][-1]:.4f}")
    print(f"  - Test Reconstruction Error: {rbm_error:.6f}")
    
    print("\n" + "=" * 70)
    print("All models trained successfully! Results saved to 'results/' directory")
    print("=" * 70)


if __name__ == "__main__":
    main()
