"""
Utility functions for deep learning models
"""
import numpy as np
from typing import Tuple, Union


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of tanh."""
    return 1 - np.tanh(x) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cross-entropy loss for classification.
    
    Args:
        y_true: One-hot encoded true labels (N, num_classes)
        y_pred: Predicted probabilities from softmax (N, num_classes)
    
    Returns:
        Average cross-entropy loss
    """
    m = y_true.shape[0]
    log_likelihood = -np.log(np.clip(np.sum(y_true * y_pred, axis=1), 1e-12, 1))
    return np.sum(log_likelihood) / m


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss."""
    return np.mean((y_true - y_pred) ** 2)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy for classification.
    
    Args:
        y_true: One-hot encoded true labels
        y_pred: Predicted probabilities
    
    Returns:
        Accuracy percentage
    """
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_true, axis=1)
    return np.mean(pred_labels == true_labels) * 100


def initialize_weights(shape: Tuple, method: str = 'xavier') -> np.ndarray:
    """
    Initialize weights for neural networks.
    
    Args:
        shape: Shape of weight matrix
        method: 'xavier' or 'he' initialization
    
    Returns:
        Initialized weight matrix
    """
    if method == 'xavier':
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return np.random.uniform(-limit, limit, shape)
    elif method == 'he':
        return np.random.randn(*shape) * np.sqrt(2 / shape[0])
    else:
        return np.random.randn(*shape) * 0.01


def initialize_bias(shape: Tuple) -> np.ndarray:
    """Initialize bias terms to zero."""
    return np.zeros(shape)


def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int):
    """
    Generate batches for training.
    
    Args:
        X: Input features
        y: Target labels
        batch_size: Number of samples per batch
    
    Yields:
        (X_batch, y_batch)
    """
    m = X.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    for start_idx in range(0, m, batch_size):
        end_idx = min(start_idx + batch_size, m)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def normalize(X: np.ndarray, axis: Union[int, Tuple] = None) -> Tuple[np.ndarray, Tuple]:
    """
    Normalize input data to [0, 1] range.
    
    Args:
        X: Input data
        axis: Axis along which to compute min/max
    
    Returns:
        Normalized data and (min, max) tuple for denormalization
    """
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)
    X_normalized = (X - x_min) / (x_max - x_min + 1e-8)
    return X_normalized, (x_min, x_max)


def denormalize(X: np.ndarray, min_max: Tuple) -> np.ndarray:
    """Denormalize data back to original range."""
    x_min, x_max = min_max
    return X * (x_max - x_min) + x_min


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot encoding.
    
    Args:
        y: Class labels (1D array)
        num_classes: Number of classes
    
    Returns:
        One-hot encoded matrix
    """
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y.astype(int)] = 1
    return one_hot


def decay_learning_rate(initial_lr: float, epoch: int, decay_rate: float) -> float:
    """
    Compute decayed learning rate.
    
    Args:
        initial_lr: Initial learning rate
        epoch: Current epoch
        decay_rate: Decay rate
    
    Returns:
        Decayed learning rate
    """
    return initial_lr / (1 + decay_rate * epoch)
