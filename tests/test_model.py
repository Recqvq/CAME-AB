import pytest
import torch
import numpy as np
from src.models import EnhancedMultiModalTransformer
from src.data import MultiModalDataset, normalize_features


def test_model_creation():
    """Test model initialization."""
    model = EnhancedMultiModalTransformer(
        embed_dim=256,
        num_heads=8,
        num_layers=2,
        num_classes=10,
        dropout=0.1
    )
    assert model is not None
    
    # Test forward pass
    batch_size = 4
    num_modalities = 5
    feature_dim = 256
    x = torch.randn(batch_size, num_modalities, feature_dim)
    
    output, modal_outputs, attn_weights = model(x)
    assert output.shape == (batch_size, 10)
    assert len(modal_outputs) == 5


def test_dataset():
    """Test dataset creation and loading."""
    # Create dummy data
    features = np.random.randn(100, 5, 256).astype(np.float32)
    labels = np.random.randint(0, 10, 100)
    
    dataset = MultiModalDataset(features, labels, augment=False)
    assert len(dataset) == 100
    
    x, y = dataset[0]
    assert x.shape == (5, 256)
    assert isinstance(y.item(), int)


def test_normalize_features():
    """Test feature normalization."""
    features = np.random.randn(100, 256)
    normalized = normalize_features(features)
    
    # Check that mean is close to 0 and std is close to 1
    assert np.abs(normalized.mean()) < 0.1
    assert np.abs(normalized.std() - 1.0) < 0.1