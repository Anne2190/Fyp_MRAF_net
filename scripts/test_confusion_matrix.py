import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import compute_confusion_matrix

def test_confusion_matrix():
    print("Testing Confusion Matrix Computation...")
    
    # Create dummy data: 4 classes (0, 1, 2, 3)
    # Shape (32, 32, 32)
    shape = (32, 32, 32)
    target = np.zeros(shape, dtype=np.int64)
    pred = np.zeros(shape, dtype=np.int64)
    
    # Class 1: (5, 10, 5:10, 5:10)
    target[5:10, 5:10, 5:10] = 1
    pred[6:11, 5:10, 5:10] = 1
    
    # Class 2
    target[15:20, 15:20, 15:20] = 2
    pred[15:20, 15:20, 15:20] = 2
    pred[14:16, 15:20, 15:20] = 2 # Some overlap
    
    # Class 3
    target[25:30, 25:30, 25:30] = 3
    pred[26:31, 26:31, 26:31] = 3
    
    cm = compute_confusion_matrix(pred, target, num_classes=4)
    
    print("Confusion Matrix:")
    print(cm)
    
    # Check shape
    assert cm.shape == (4, 4), f"Wrong shape: {cm.shape}"
    
    # Check that sum is equal to number of voxels
    assert np.sum(cm) == np.prod(shape), f"Sum {np.sum(cm)} != {np.prod(shape)}"
    
    print("\n✓ Confusion Matrix test passed!")

if __name__ == '__main__':
    test_confusion_matrix()
