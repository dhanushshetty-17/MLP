"""
Multilayer Perceptron (MLP) classifier from scratch using NumPy
"""
import numpy as np
from utils import (
    relu, relu_derivative, sigmoid, softmax, cross_entropy_loss,
    initialize_weights, initialize_bias, batch_generator
)
from typing import Tuple, List


class MLP:
    """
    Two-layer Multilayer Perceptron for classification.
    
    Architecture: Input -> Hidden -> Output
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 hidden_activation: str = 'relu', learning_rate: float = 0.05):
        """
        Initialize MLP.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output classes
            hidden_activation: 'relu' or 'sigmoid'
            learning_rate: Learning rate for SGD
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        
        # Initialize weights
        self.W1 = initialize_weights((input_size, hidden_size), 'xavier')
        self.b1 = initialize_bias((hidden_size,))
        self.W2 = initialize_weights((hidden_size, output_size), 'xavier')
        self.b2 = initialize_bias((output_size,))
        
        # Store activations for backprop
        self.cache = {}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation.
        
        Args:
            X: Input data (batch_size, input_size)
        
        Returns:
            Output probabilities (batch_size, output_size)
        """
        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        
        if self.hidden_activation == 'relu':
            self.a1 = relu(self.z1)
        else:  # sigmoid
            self.a1 = sigmoid(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        
        # Cache for backward pass
        self.cache['X'] = X
        self.cache['z1'] = self.z1
        self.cache['a1'] = self.a1
        
        return self.a2
    
    def backward(self, y_true: np.ndarray) -> None:
        """
        Backpropagation.
        
        Args:
            y_true: True labels (one-hot encoded)
        """
        m = y_true.shape[0]
        X = self.cache['X']
        a1 = self.cache['a1']
        
        # Output layer gradient
        dz2 = self.a2 - y_true
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0) / m
        
        # Hidden layer gradient
        da1 = np.dot(dz2, self.W2.T)
        
        if self.hidden_activation == 'relu':
            dz1 = da1 * relu_derivative(self.cache['z1'])
        else:  # sigmoid
            dz1 = da1 * sigmoid(self.cache['z1']) * (1 - sigmoid(self.cache['z1']))
        
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 30,
              batch_size: int = 64, validation_data: Tuple = None,
              verbose: bool = True) -> dict:
        """
        Train the MLP using mini-batch SGD.
        
        Args:
            X: Training features
            y: Training labels (one-hot encoded)
            epochs: Number of epochs
            batch_size: Batch size for SGD
            validation_data: (X_val, y_val) for validation
            verbose: Print training progress
        
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            # Training loop
            for X_batch, y_batch in batch_generator(X, y, batch_size):
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                batch_loss = cross_entropy_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Compute accuracy
                from utils import accuracy
                batch_acc = accuracy(y_batch, y_pred)
                epoch_acc += batch_acc
                
                # Backward pass
                self.backward(y_batch)
                
                n_batches += 1
            
            # Average metrics
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            # Validation
            if validation_data is not None:
                X_val, y_val = validation_data
                y_val_pred = self.forward(X_val)
                val_loss = cross_entropy_loss(y_val, y_val_pred)
                from utils import accuracy
                val_acc = accuracy(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input data
        
        Returns:
            Predicted class labels
        """
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input data
        
        Returns:
            Class probabilities
        """
        return self.forward(X)
