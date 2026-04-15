"""
Restricted Boltzmann Machine (RBM) from scratch using NumPy
Implements Contrastive Divergence learning algorithm
"""
import numpy as np
from utils import sigmoid, initialize_weights, batch_generator
from typing import Tuple


class RBM:
    """
    Restricted Boltzmann Machine for unsupervised feature learning.
    
    Uses Contrastive Divergence (CD-1) for training.
    """
    
    def __init__(self, n_visible: int, n_hidden: int, learning_rate: float = 0.01):
        """
        Initialize RBM.
        
        Args:
            n_visible: Number of visible units (input dimension)
            n_hidden: Number of hidden units
            learning_rate: Learning rate for weight updates
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W = np.random.randn(n_visible, n_hidden) * 0.01
        self.bv = np.zeros(n_visible)  # Visible bias
        self.bh = np.zeros(n_hidden)   # Hidden bias
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sample_h_given_v(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample hidden units given visible units (bottom-up).
        
        Args:
            v: Visible units (batch_size, n_visible)
        
        Returns:
            (h_probabilities, h_samples)
        """
        h_prob = self.sigmoid(np.dot(v, self.W) + self.bh)
        h_sample = (np.random.uniform(0, 1, h_prob.shape) < h_prob).astype(float)
        return h_prob, h_sample
    
    def sample_v_given_h(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample visible units given hidden units (top-down).
        
        Args:
            h: Hidden units (batch_size, n_hidden)
        
        Returns:
            (v_probabilities, v_samples)
        """
        v_prob = self.sigmoid(np.dot(h, self.W.T) + self.bv)
        v_sample = (np.random.uniform(0, 1, v_prob.shape) < v_prob).astype(float)
        return v_prob, v_sample
    
    def contrastive_divergence(self, v: np.ndarray, k: int = 1) -> Tuple:
        """
        Contrastive Divergence algorithm (CD-k).
        
        Args:
            v: Visible units (training batch)
            k: Number of Gibbs sampling steps
        
        Returns:
            (v_data, h_data, v_model, h_model)
        """
        # Positive phase (data-dependent)
        h_prob_data, h_data = self.sample_h_given_v(v)
        
        # Negative phase (model-dependent) - perform k steps of Gibbs sampling
        v_model = v
        h_model = h_data
        
        for _ in range(k):
            v_prob_model, v_model = self.sample_v_given_h(h_model)
            h_prob_model, h_model = self.sample_h_given_v(v_model)
        
        return v, h_prob_data, v_model, h_prob_model
    
    def train(self, X: np.ndarray, epochs: int = 30, batch_size: int = 64,
              cd_k: int = 1, verbose: bool = True) -> dict:
        """
        Train RBM using Contrastive Divergence.
        
        Args:
            X: Training data (batch_size, n_visible)
            epochs: Number of training epochs
            batch_size: Batch size for SGD
            cd_k: Number of Gibbs sampling steps
            verbose: Print progress
        
        Returns:
            Training history
        """
        history = {'free_energy': []}
        
        for epoch in range(epochs):
            epoch_energy = 0
            n_batches = 0
            
            for X_batch, _ in batch_generator(X, X, batch_size):
                # Contrastive Divergence
                v_data, h_data, v_model, h_model = self.contrastive_divergence(X_batch, cd_k)
                
                # Update weights
                m = X_batch.shape[0]
                dW = np.dot(v_data.T, h_data) - np.dot(v_model.T, h_model)
                dW /= m
                
                # Update biases
                dv = np.sum(v_data - v_model, axis=0) / m
                dh = np.sum(h_data - h_model, axis=0) / m
                
                # Apply updates
                self.W += self.learning_rate * dW
                self.bv += self.learning_rate * dv
                self.bh += self.learning_rate * dh
                
                # Compute free energy (for monitoring)
                energy = -np.sum(np.dot(X_batch, self.bv)) - np.sum(
                    np.log(1 + np.exp(np.dot(X_batch, self.W) + self.bh))
                )
                epoch_energy += energy / m
                n_batches += 1
            
            avg_energy = epoch_energy / n_batches
            history['free_energy'].append(avg_energy)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Free Energy: {avg_energy:.4f}")
        
        return history
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Get hidden unit activations (features) for input data.
        
        Args:
            X: Input data
        
        Returns:
            Hidden unit probabilities
        """
        h_prob, _ = self.sample_h_given_v(X)
        return h_prob
    
    def reconstruct(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Reconstruct input by sampling from RBM.
        
        Args:
            X: Input data
            steps: Number of Gibbs sampling steps
        
        Returns:
            Reconstructed data
        """
        v = X
        for _ in range(steps):
            h_prob, h_sample = self.sample_h_given_v(v)
            v_prob, v_sample = self.sample_v_given_h(h_sample)
            v = v_sample
        
        return v
    
    def get_reconstruction_error(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Get reconstruction error for each sample.
        
        Args:
            X: Input data
            steps: Number of Gibbs sampling steps
        
        Returns:
            Reconstruction error per sample
        """
        X_reconstructed = self.reconstruct(X, steps)
        return np.mean(np.abs(X - X_reconstructed), axis=1)
    
    def get_weights_visualization(self) -> np.ndarray:
        """
        Get weight matrix for visualization.
        
        Returns:
            Weight matrix
        """
        return self.W
