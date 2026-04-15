"""
Quick test script to verify all components work
"""
import sys
import os

print("=" * 70)
print("QUICK COMPONENT TEST")
print("=" * 70)

# Test 1: Imports
print("\n[Test 1] Testing imports...")
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from utils import relu, sigmoid, softmax, cross_entropy_loss, one_hot_encode
    from data_loader import FashionMNISTLoader
    from mlp import MLP
    from autoencoder import DenseAutoencoder
    from rbm import RBM
    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Utility Functions
print("\n[Test 2] Testing utility functions...")
try:
    x = np.array([[-1, 0, 1, 2]])
    assert relu(x).shape == x.shape
    assert sigmoid(x).shape == x.shape
    
    y = np.array([[0.1, 0.2, 0.7]])
    sm = softmax(y)
    assert np.abs(np.sum(sm) - 1.0) < 1e-6
    
    y_true = one_hot_encode(np.array([0, 1, 2]), 3)
    assert y_true.shape == (3, 3)
    
    print("  [OK] Utility functions work correctly")
except Exception as e:
    print(f"  [FAIL] Utility error: {e}")
    sys.exit(1)

# Test 3: MLP Forward Pass
print("\n[Test 3] Testing MLP forward pass...")
try:
    mlp = MLP(input_size=10, hidden_size=5, output_size=3, learning_rate=0.01)
    X_sample = np.random.randn(2, 10)
    output = mlp.forward(X_sample)
    assert output.shape == (2, 3)
    assert np.abs(np.sum(output, axis=1) - 1.0).max() < 1e-6  # Softmax sums to 1
    print("  [OK] MLP forward pass works")
except Exception as e:
    print(f"  [FAIL] MLP error: {e}")
    sys.exit(1)

# Test 4: Autoencoder Forward Pass
print("\n[Test 4] Testing Autoencoder forward pass...")
try:
    ae = DenseAutoencoder(input_size=20, hidden_sizes=[10], latent_size=5)
    X_sample = np.random.randn(2, 20)
    output, cache = ae.forward(X_sample)
    assert output.shape == (2, 20)
    latent = ae.encode_data(X_sample)
    assert latent.shape == (2, 5)
    print("  [OK] Autoencoder forward pass works")
except Exception as e:
    print(f"  [FAIL] Autoencoder error: {e}")
    sys.exit(1)

# Test 5: RBM Sampling
print("\n[Test 5] Testing RBM sampling...")
try:
    rbm = RBM(n_visible=20, n_hidden=10, learning_rate=0.01)
    X_sample = np.random.binomial(1, 0.5, (2, 20)).astype(float)
    h_prob, h_sample = rbm.sample_h_given_v(X_sample)
    assert h_prob.shape == (2, 10)
    assert h_sample.shape == (2, 10)
    print("  [OK] RBM sampling works")
except Exception as e:
    print(f"  [FAIL] RBM error: {e}")
    sys.exit(1)

# Test 6: Data Loader (small test)
print("\n[Test 6] Testing data loader structure...")
try:
    loader = FashionMNISTLoader()
    # Just check if object is created properly
    assert hasattr(loader, 'load')
    assert hasattr(loader, 'download')
    print("  [OK] Data loader initialized")
except Exception as e:
    print(f"  [FAIL] Data loader error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nYour code is working correctly. You can now run:")
print("  c:/python314/python.exe train.py")
print("\n" + "=" * 70)
