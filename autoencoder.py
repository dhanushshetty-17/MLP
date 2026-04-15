"""
Dense Autoencoder from scratch using NumPy
Supports both undercomplete and sparse variants
"""
import numpy as np
from utils import (
    sigmoid, relu, relu_derivative, mse_loss,
    initialize_weights, initialize_bias, batch_generator
)
from typing import Tuple, List


class DenseAutoencoder:
    """
    Dense Autoencoder for unsupervised feature learning and reconstruction.
    
    Architecture: Input -> Encoder -> Latent -> Decoder -> Output
    """
    
    def __init__(self, input_size: int, hidden_sizes: List[int],
                 latent_size: int, learning_rate: float = 0.01,
                 activation: str = 'relu', sparse: bool = False,
                 sparsity_weight: float = 0.01):
        """
        Initialize Autoencoder.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes in encoder
            latent_size: Size of latent representation
            learning_rate: Learning rate for SGD
            activation: 'relu' or 'sigmoid'
            sparse: Whether to use sparse autoencoder
            sparsity_weight: Weight for sparsity regularization
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.sparse = sparse
        self.sparsity_weight = sparsity_weight
        
        # Build encoder
        self.encoder_weights = []
        self.encoder_biases = []
        
        layer_sizes = [input_size] + hidden_sizes + [latent_size]
        for i in range(len(layer_sizes) - 1):
            W = initialize_weights((layer_sizes[i], layer_sizes[i+1]), 'xavier')
            b = initialize_bias((layer_sizes[i+1],))
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)
        
        # Build decoder (mirror of encoder)
        self.decoder_weights = []
        self.decoder_biases = []
        
        decoder_sizes = [latent_size] + hidden_sizes[::-1] + [input_size]
        for i in range(len(decoder_sizes) - 1):
            W = initialize_weights((decoder_sizes[i], decoder_sizes[i+1]), 'xavier')
            b = initialize_bias((decoder_sizes[i+1],))
            self.decoder_weights.append(W)
            self.decoder_biases.append(b)
        
        self.cache = {}
    
    def encode(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Encode input to latent representation.
        
        Args:
            X: Input data
        
        Returns:
            (latent_representation, cache_for_backprop)
        """
        cache = {
            'encoder_activations': [X],
            'encoder_z': []  # Store pre-activation values
        }
        
        # Forward through encoder
        a = X
        for i, (W, b) in enumerate(zip(self.encoder_weights, self.encoder_biases)):
            z = np.dot(a, W) + b
            cache['encoder_z'].append(z)
            
            if i < len(self.encoder_weights) - 1:  # Hidden layers
                if self.activation == 'relu':
                    a = relu(z)
                else:  # sigmoid
                    a = sigmoid(z)
            else:  # Latent layer
                a = sigmoid(z)  # Always sigmoid for latent to keep in [0, 1]
            
            cache['encoder_activations'].append(a)
        
        latent = a
        if self.sparse:
            cache['latent_activations'] = latent
        
        return latent, cache
    
    def decode(self, latent: np.ndarray, cache: dict) -> Tuple[np.ndarray, dict]:
        """
        Decode latent representation back to input space.
        
        Args:
            latent: Latent representation
            cache: Cache from encoder
        
        Returns:
            (reconstructed_output, updated_cache)
        """
        a = latent
        cache['decoder_activations'] = [latent]
        cache['decoder_z'] = []  # Store pre-activation values
        
        # Forward through decoder
        for i, (W, b) in enumerate(zip(self.decoder_weights, self.decoder_biases)):
            z = np.dot(a, W) + b
            cache['decoder_z'].append(z)
            
            if i < len(self.decoder_weights) - 1:  # Hidden layers
                if self.activation == 'relu':
                    a = relu(z)
                else:  # sigmoid
                    a = sigmoid(z)
            else:  # Output layer
                a = sigmoid(z)  # Sigmoid for normalized output
            
            cache['decoder_activations'].append(a)
        
        return a, cache
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Forward pass through entire autoencoder.
        
        Args:
            X: Input data
        
        Returns:
            (reconstructed_output, cache)
        """
        latent, cache = self.encode(X)
        output, cache = self.decode(latent, cache)
        return output, cache
    
    def backward(self, X: np.ndarray, output: np.ndarray, cache: dict) -> None:
        """
        Backpropagation through autoencoder.
        
        Args:
            X: Original input
            output: Reconstructed output
            cache: Cache from forward pass
        """
        m = X.shape[0]
        
        # Output layer gradient (MSE loss derivative)
        delta = 2.0 * (output - X) / m
        
        # Apply sigmoid derivative for output layer
        delta = delta * output * (1 - output)
        
        # Backprop through decoder
        for i in range(len(self.decoder_weights) - 1, -1, -1):
            W = self.decoder_weights[i]
            
            # Gradient update
            dW = np.dot(cache['decoder_activations'][i].T, delta)
            db = np.sum(delta, axis=0)
            
            # Clip gradients to prevent explosion
            dW = np.clip(dW, -5, 5)
            db = np.clip(db, -5, 5)
            
            # Update weights
            self.decoder_weights[i] -= self.learning_rate * dW
            self.decoder_biases[i] -= self.learning_rate * db
            
            # Propagate delta to previous layer
            if i > 0:
                delta = np.dot(delta, W.T)
                # Apply activation derivative for hidden layers
                z = cache['decoder_z'][i-1]
                if self.activation == 'relu':
                    delta *= relu_derivative(z)
                else:  # sigmoid
                    a = cache['decoder_activations'][i]
                    delta *= a * (1 - a)
            else:
                # Propagate to encoder
                delta = np.dot(delta, W.T)
        
        # Backprop through encoder  
        for i in range(len(self.encoder_weights) - 1, -1, -1):
            W = self.encoder_weights[i]
            
            # Gradient update
            dW = np.dot(cache['encoder_activations'][i].T, delta)
            db = np.sum(delta, axis=0)
            
            # Add sparsity gradient if enabled
            if self.sparse and i == len(self.encoder_weights) - 1:
                target_sparsity = 0.05
                actual_sparsity = np.mean(cache['latent_activations'], axis=0)
                sparsity_grad = self.sparsity_weight * (actual_sparsity - target_sparsity)
                dW += cache['encoder_activations'][i].T @ np.tile(sparsity_grad, (m, 1))
            
            # Clip gradients to prevent explosion
            dW = np.clip(dW, -5, 5)
            db = np.clip(db, -5, 5)
            
            # Update weights
            self.encoder_weights[i] -= self.learning_rate * dW
            self.encoder_biases[i] -= self.learning_rate * db
            
            # Propagate delta to previous layer
            if i > 0:
                delta = np.dot(delta, W.T)
                z = cache['encoder_z'][i-1]
                if self.activation == 'relu':
                    delta *= relu_derivative(z)
                else:  # sigmoid
                    a = cache['encoder_activations'][i]
                    delta *= a * (1 - a)
    
    def train(self, X: np.ndarray, epochs: int = 30, batch_size: int = 64,
              validation_data: np.ndarray = None, verbose: bool = True) -> dict:
        """
        Train the autoencoder.
        
        Args:
            X: Training data
            epochs: Number of epochs
            batch_size: Batch size
            validation_data: Validation data for monitoring
            verbose: Print progress
        
        Returns:
            Training history
        """
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            for X_batch, _ in batch_generator(X, X, batch_size):
                # Forward pass
                output, cache = self.forward(X_batch)
                
                # Check for NaN in output
                if np.isnan(output).any():
                    print(f"\n[ERROR] NaN detected in output at epoch {epoch+1}, batch {n_batches+1}")
                    print("This usually indicates:")
                    print("  1. Learning rate too high (try reducing it)")
                    print("  2. Numerical instability in activations")
                    print("  3. Exploding gradients")
                    raise ValueError("NaN detected in forward pass")
                
                # Compute loss
                batch_loss = mse_loss(X_batch, output)
                
                # Check for NaN in loss
                if np.isnan(batch_loss):
                    print(f"\n[ERROR] NaN detected in loss at epoch {epoch+1}, batch {n_batches+1}")
                    raise ValueError("NaN detected in loss computation")
                
                epoch_loss += batch_loss
                
                # Backward pass
                self.backward(X_batch, output, cache)
                
                n_batches += 1
            
            epoch_loss /= n_batches
            history['train_loss'].append(epoch_loss)
            
            # Validation
            if validation_data is not None:
                val_output, _ = self.forward(validation_data)
                val_loss = mse_loss(validation_data, val_output)
                history['val_loss'].append(val_loss)
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.6f}")
        
        return history
    
    def encode_data(self, X: np.ndarray) -> np.ndarray:
        """Get latent representations for data."""
        latent, _ = self.encode(X)
        return latent
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data."""
        output, _ = self.forward(X)
        return output
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction error for each sample."""
        output, _ = self.forward(X)
        return np.mean((X - output) ** 2, axis=1)
