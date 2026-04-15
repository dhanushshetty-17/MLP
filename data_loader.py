"""
Data loader for Fashion-MNIST dataset
"""
import numpy as np
import urllib.request
import gzip
import os
from typing import Tuple


class FashionMNISTLoader:
    """Loads and preprocesses Fashion-MNIST dataset."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.url_base = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        self.files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
    
    def download(self):
        """Download Fashion-MNIST dataset."""
        print("Downloading Fashion-MNIST dataset...")
        for file_type, file_name in self.files.items():
            file_path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(file_path):
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(
                    self.url_base + file_name,
                    file_path
                )
                print(f"  [OK] {file_name}")
        print("Download complete!")
    
    def load_images(self, file_path: str) -> np.ndarray:
        """Load images from gzip file."""
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28 * 28)
        return data.astype(np.float32) / 255.0
    
    def load_labels(self, file_path: str) -> np.ndarray:
        """Load labels from gzip file."""
        with gzip.open(file_path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    def load(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load and return the Fashion-MNIST dataset.
        
        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        # Download if not already present
        if not all(os.path.exists(os.path.join(self.data_dir, f)) 
                   for f in self.files.values()):
            self.download()
        
        # Load training data
        X_train = self.load_images(os.path.join(self.data_dir, self.files['train_images']))
        y_train = self.load_labels(os.path.join(self.data_dir, self.files['train_labels']))
        
        # Load test data
        X_test = self.load_images(os.path.join(self.data_dir, self.files['test_images']))
        y_test = self.load_labels(os.path.join(self.data_dir, self.files['test_labels']))
        
        return (X_train, y_train), (X_test, y_test)
    
    def create_validation_split(self, X_train: np.ndarray, y_train: np.ndarray, 
                                validation_split: float = 0.2) -> Tuple[Tuple, Tuple]:
        """
        Split training data into training and validation sets.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_split: Fraction of data for validation
        
        Returns:
            ((X_tr, y_tr), (X_val, y_val))
        """
        n_samples = X_train.shape[0]
        n_validation = int(n_samples * validation_split)
        
        # Shuffle indices
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_validation]
        train_indices = indices[n_validation:]
        
        X_tr = X_train[train_indices]
        y_tr = y_train[train_indices]
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        
        return (X_tr, y_tr), (X_val, y_val)


def get_fashion_mnist(validation_split: float = 0.2):
    """
    Convenience function to load Fashion-MNIST with train/val/test split.
    
    Args:
        validation_split: Fraction of training data for validation
    
    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    loader = FashionMNISTLoader()
    (X_train, y_train), (X_test, y_test) = loader.load()
    
    (X_train, y_train), (X_val, y_val) = loader.create_validation_split(
        X_train, y_train, validation_split
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
