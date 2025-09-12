import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
import random
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """
    Custom dataset for multi-modal data with optional augmentation.
    
    Args:
        features: Input features array
        labels: Target labels array
        augment: Whether to apply data augmentation
        augment_prob: Probability of applying augmentation
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 augment: bool = False, augment_prob: float = 0.3):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.augment_prob = augment_prob
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx].clone()
        y = self.labels[idx]
        
        if self.augment and random.random() < self.augment_prob:
            mask_percent = random.uniform(0.1, 0.3)
            feature_size = x.shape[1]
            num_masked = int(feature_size * mask_percent)
            
            for modality_idx in range(x.shape[0]):
                mask_indices = torch.randperm(feature_size)[:num_masked]
                if random.random() < 0.5:
                    x[modality_idx, mask_indices] = 0
                else:
                    noise = torch.randn(num_masked) * 0.1
                    x[modality_idx, mask_indices] += noise
                    
        return x, y


def create_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """
    Create a weighted sampler for imbalanced datasets.
    
    Args:
        labels: Array of labels
        
    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def load_features(config: Dict) -> Dict[str, np.ndarray]:
    """
    Load features from specified paths in configuration.
    
    Args:
        config: Configuration dictionary containing file paths
        
    Returns:
        Dictionary containing loaded features and labels
    """
    features_dict = {}
    
    try:
        # Load ESMC features
        features_dict['esmc_train'] = np.load(config['paths']['esmc']['train_features'])
        features_dict['esmc_train_labels'] = np.load(config['paths']['esmc']['train_labels'])
        features_dict['esmc_val'] = np.load(config['paths']['esmc']['val_features'])
        features_dict['esmc_val_labels'] = np.load(config['paths']['esmc']['val_labels'])
        features_dict['esmc_test'] = np.load(config['paths']['esmc']['test_features'])
        features_dict['esmc_test_labels'] = np.load(config['paths']['esmc']['test_labels'])
        
        # Load ESMCStructure features
        features_dict['esmcStructure_train'] = np.load(config['paths']['esmcStructure']['train_features'])
        features_dict['esmcStructure_train_labels'] = np.load(config['paths']['esmcStructure']['train_labels'])
        features_dict['esmcStructure_val'] = np.load(config['paths']['esmcStructure']['val_features'])
        features_dict['esmcStructure_val_labels'] = np.load(config['paths']['esmcStructure']['val_labels'])
        features_dict['esmcStructure_test'] = np.load(config['paths']['esmcStructure']['test_features'])
        features_dict['esmcStructure_test_labels'] = np.load(config['paths']['esmcStructure']['test_labels'])
        
        # Load OneHot features
        features_dict['onehot_train'] = np.load(config['paths']['onehot']['train_features'])
        features_dict['onehot_train_labels'] = np.load(config['paths']['onehot']['train_labels'])
        features_dict['onehot_val'] = np.load(config['paths']['onehot']['val_features'])
        features_dict['onehot_val_labels'] = np.load(config['paths']['onehot']['val_labels'])
        features_dict['onehot_test'] = np.load(config['paths']['onehot']['test_features'])
        features_dict['onehot_test_labels'] = np.load(config['paths']['onehot']['test_labels'])
        
        # Load BLOSUM features
        features_dict['blosum_train'] = np.load(config['paths']['blosum']['train_features'])
        features_dict['blosum_train_labels'] = np.load(config['paths']['blosum']['train_labels'])
        features_dict['blosum_val'] = np.load(config['paths']['blosum']['val_features'])
        features_dict['blosum_val_labels'] = np.load(config['paths']['blosum']['val_labels'])
        features_dict['blosum_test'] = np.load(config['paths']['blosum']['test_features'])
        features_dict['blosum_test_labels'] = np.load(config['paths']['blosum']['test_labels'])
        
        # Load GCN features
        features_dict['gcn_train_val'] = np.load(config['paths']['gcn']['train_val_features'])
        features_dict['gcn_train_val_labels'] = np.load(config['paths']['gcn']['train_val_labels'])
        features_dict['gcn_test'] = np.load(config['paths']['gcn']['test_features'])
        features_dict['gcn_test_labels'] = np.load(config['paths']['gcn']['test_labels'])
        
        logger.info("Successfully loaded all feature sets")
        return features_dict
        
    except FileNotFoundError as e:
        logger.error(f"Error loading features: {e}")
        raise


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: Input features array
        
    Returns:
        Normalized features
    """
    scaler = StandardScaler()
    return scaler.fit_transform(features)


def create_modality_tensors(esmc: np.ndarray, 
                           esmcStructure: np.ndarray, 
                           onehot: np.ndarray, 
                           blosum: np.ndarray, 
                           gcn: np.ndarray) -> np.ndarray:
    """
    Create modality tensors from individual feature arrays.
    
    Args:
        esmc: ESMC features
        esmcStructure: ESMC Structure features
        onehot: OneHot features
        blosum: BLOSUM features
        gcn: GCN features
        
    Returns:
        Combined modality tensor
    """
    batch_size = esmc.shape[0]
    modality_tensors = np.zeros((batch_size, 5, 256), dtype=np.float32)
    
    modality_tensors[:, 0, :] = esmc
    modality_tensors[:, 1, :] = esmcStructure
    modality_tensors[:, 2, :] = onehot
    modality_tensors[:, 3, :] = blosum
    modality_tensors[:, 4, :] = gcn
    
    return modality_tensors


def feature_mixup(features: torch.Tensor, 
                  labels: torch.Tensor, 
                  alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup augmentation to features and labels.
    
    Args:
        features: Input features
        labels: Input labels
        alpha: Mixup alpha parameter
        
    Returns:
        Tuple of (mixed_features, labels_a, labels_b, lambda)
    """
    batch_size = features.shape[0]
    indices = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)
    
    mixed_features = lam * features + (1 - lam) * features[indices]
    
    return mixed_features, labels, labels[indices], lam