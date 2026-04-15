"""
Evaluation and analysis script for trained models
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_fashion_mnist
from utils import one_hot_encode, accuracy as compute_accuracy
from mlp import MLP
from autoencoder import DenseAutoencoder
from rbm import RBM
from sklearn.metrics import confusion_matrix, classification_report


def load_trained_models():
    """Load all trained models."""
    # Load data first
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_fashion_mnist(validation_split=0.2)
    
    # MLP
    mlp = MLP(
        input_size=784,
        hidden_size=128,
        output_size=10,
        hidden_activation='relu',
        learning_rate=0.05
    )
    y_train_onehot = one_hot_encode(y_train, 10)
    mlp.train(X_train, y_train_onehot, epochs=30, batch_size=64, 
              validation_data=(X_val, one_hot_encode(y_val, 10)), verbose=False)
    
    # Autoencoder
    autoencoder = DenseAutoencoder(
        input_size=784,
        hidden_sizes=[256],
        latent_size=32,
        learning_rate=0.01,
        activation='relu'
    )
    autoencoder.train(X_train, epochs=30, batch_size=64, 
                     validation_data=X_val, verbose=False)
    
    # RBM
    rbm = RBM(n_visible=784, n_hidden=100, learning_rate=0.01)
    rbm.train(X_train, epochs=30, batch_size=64, verbose=False)
    
    return mlp, autoencoder, rbm, (X_test, y_test)


def analyze_mlp(mlp, X_test, y_test):
    """Analyze MLP performance."""
    print("\n" + "="*70)
    print("MLP CLASSIFICATION ANALYSIS")
    print("="*70)
    
    # Predictions
    y_pred = mlp.predict(X_test)
    y_test_onehot = one_hot_encode(y_test, 10)
    y_pred_onehot = one_hot_encode(y_pred, 10)
    
    # Accuracy
    test_acc = compute_accuracy(y_test_onehot, mlp.predict_proba(X_test))
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    
    # Per-class accuracy
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("\nPer-Class Performance:")
    for i, class_name in enumerate(class_names):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == y_test[mask]) * 100
            print(f"  {class_name:15s}: {class_acc:6.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Manual heatmap without seaborn
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('MLP Confusion Matrix')
    plt.tight_layout()
    plt.savefig('results/mlp_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Confusion matrix saved")
    plt.close()
    
    return test_acc


def analyze_autoencoder(autoencoder, X_test):
    """Analyze Autoencoder performance."""
    print("\n" + "="*70)
    print("AUTOENCODER RECONSTRUCTION ANALYSIS")
    print("="*70)
    
    # Reconstruction error
    X_reconstructed = autoencoder.reconstruct(X_test)
    reconstruction_errors = np.mean((X_test - X_reconstructed) ** 2, axis=1)
    
    print(f"\nReconstruction Error (MSE):")
    print(f"  Mean: {np.mean(reconstruction_errors):.6f}")
    print(f"  Std:  {np.std(reconstruction_errors):.6f}")
    print(f"  Min:  {np.min(reconstruction_errors):.6f}")
    print(f"  Max:  {np.max(reconstruction_errors):.6f}")
    
    # Outlier detection
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
    outliers = reconstruction_errors > threshold
    print(f"\nOutlier Detection (threshold={threshold:.6f}):")
    print(f"  Outliers detected: {np.sum(outliers)} ({np.sum(outliers)/len(X_test)*100:.2f}%)")
    
    # Error distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(reconstruction_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.6f}')
    axes[0].set_xlabel('Reconstruction Error (MSE)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Reconstruction Errors')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Latent space distribution
    latent_features = autoencoder.encode_data(X_test)
    axes[1].scatter(latent_features[:, 0], latent_features[:, 1], alpha=0.5, s=20)
    axes[1].set_xlabel('Latent Dimension 0')
    axes[1].set_ylabel('Latent Dimension 1')
    axes[1].set_title('Latent Space Representation (2D)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/autoencoder_analysis.png', dpi=300, bbox_inches='tight')
    print("[OK] Analysis plots saved")
    plt.close()
    
    return np.mean(reconstruction_errors)


def analyze_rbm(rbm, X_test):
    """Analyze RBM performance."""
    print("\n" + "="*70)
    print("RBM FEATURE LEARNING ANALYSIS")
    print("="*70)
    
    # Reconstruction
    X_reconstructed = rbm.reconstruct(X_test, steps=5)
    reconstruction_error = np.mean(np.abs(X_test - X_reconstructed))
    
    print(f"\nReconstruction Error (MAE): {reconstruction_error:.6f}")
    
    # Feature extraction
    hidden_features = rbm.transform(X_test)
    print(f"\nHidden Feature Statistics:")
    print(f"  Mean activation: {np.mean(hidden_features):.4f}")
    print(f"  Std activation:  {np.std(hidden_features):.4f}")
    print(f"  Sparsity:        {np.mean(hidden_features < 0.1):.2%}")
    
    # Filter visualization with stats
    W = rbm.get_weights_visualization()
    print(f"\nWeight Matrix Statistics:")
    print(f"  Mean: {np.mean(W):.6f}")
    print(f"  Std:  {np.std(W):.6f}")
    print(f"  Min:  {np.min(W):.6f}")
    print(f"  Max:  {np.max(W):.6f}")
    
    # Feature activation patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hidden unit activation distribution
    axes[0].hist(hidden_features.flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Hidden Unit Activation')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Hidden Unit Activations')
    axes[0].grid(True, alpha=0.3)
    
    # Mean activation per hidden unit
    mean_activations = np.mean(hidden_features, axis=0)
    axes[1].bar(range(min(50, len(mean_activations))), mean_activations[:50])
    axes[1].set_xlabel('Hidden Unit Index')
    axes[1].set_ylabel('Mean Activation')
    axes[1].set_title('Mean Activation per Hidden Unit (First 50)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/rbm_analysis.png', dpi=300, bbox_inches='tight')
    print("[OK] Analysis plots saved")
    plt.close()
    
    return reconstruction_error


def generate_report():
    """Generate comprehensive evaluation report."""
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS - COMPREHENSIVE EVALUATION")
    print("="*70)
    print("Dataset: Fashion-MNIST (10 classes, 28x28 grayscale images)")
    print("="*70)
    
    # Load models
    print("\nLoading trained models...")
    mlp, autoencoder, rbm, (X_test, y_test) = load_trained_models()
    
    # Analyze each model
    mlp_acc = analyze_mlp(mlp, X_test, y_test)
    ae_mse = analyze_autoencoder(autoencoder, X_test)
    rbm_error = analyze_rbm(rbm, X_test)
    
    # Summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"\n1. MLP Classification")
    print(f"   - Test Accuracy: {mlp_acc:.2f}%")
    print(f"   - Architecture: 784 → 128 (ReLU) → 10 (Softmax)")
    print(f"   - Optimizer: SGD (lr=0.05)")
    
    print(f"\n2. Autoencoder Reconstruction")
    print(f"   - Test MSE: {ae_mse:.6f}")
    print(f"   - Architecture: 784 → 256 → 32 → 256 → 784")
    print(f"   - Optimizer: SGD (lr=0.01)")
    
    print(f"\n3. RBM Feature Learning")
    print(f"   - Test Reconstruction Error: {rbm_error:.6f}")
    print(f"   - Architecture: 784 visible units, 100 hidden units")
    print(f"   - Learning: Contrastive Divergence (CD-1)")
    
    print("\n" + "="*70)
    print("All evaluation results saved to 'results/' directory")
    print("="*70)


if __name__ == "__main__":
    generate_report()
