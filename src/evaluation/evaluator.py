import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, roc_auc_score
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, Tuple, List, Optional
import os

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_probs: Optional[np.ndarray] = None,
                     num_classes: int = None) -> Dict:
    """
    Calculate various evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
        num_classes: Number of classes
        
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    if num_classes:
        target_names = [f"Class {i}" for i in range(num_classes)]
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
        metrics['classification_report'] = pd.DataFrame(report).transpose()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC-AUC if probabilities are provided
    if y_probs is not None and num_classes and num_classes > 2:
        try:
            # One-vs-rest ROC-AUC for multiclass
            metrics['roc_auc'] = {}
            for i in range(num_classes):
                binary_true = (y_true == i).astype(int)
                if len(np.unique(binary_true)) > 1:  # Check if both classes exist
                    class_probs = y_probs[:, i]
                    metrics['roc_auc'][f'class_{i}'] = roc_auc_score(binary_true, class_probs)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    return metrics


def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader,
                  criterion: nn.Module,
                  device: torch.device,
                  num_classes: int,
                  save_path: Optional[str] = None) -> Dict:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run on
        num_classes: Number of output classes
        save_path: Optional path to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Starting model evaluation...")
    model.eval()
    
    test_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    all_attention_weights = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating model"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, modal_outputs, modal_projections, global_projection, attention_weights = model(
                inputs, return_features=True
            )
            
            # Calculate loss
            loss, _, _, _ = criterion(
                outputs, modal_outputs, modal_projections, global_projection,
                targets, 0, 1
            )
            
            test_loss += loss.item() * inputs.size(0)
            
            # Get predictions and probabilities
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs) if all_probs else np.array([])
    all_attention_weights = np.vstack(all_attention_weights)
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    metrics = calculate_metrics(all_targets, all_preds, all_probs, num_classes)
    metrics['test_loss'] = test_loss
    
    # Calculate average attention weights
    avg_attention = all_attention_weights.mean(axis=0)
    modality_names = ['ESMC', 'ESMCStructure', 'OneHot', 'BLOSUM', 'GCN']
    metrics['modality_importance'] = dict(zip(modality_names, avg_attention.squeeze()))
    
    # Log results
    logger.info(f"\nTest Results:")
    logger.info(f"Loss: {test_loss:.4f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Micro F1: {metrics['micro_f1']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    logger.info(f"Modality Importance: {metrics['modality_importance']}")
    
    # Save results if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        
        # Save classification report
        if 'classification_report' in metrics:
            metrics['classification_report'].to_csv(
                f"{save_path}/classification_report.csv"
            )
        
        # Save confusion matrix
        np.save(f"{save_path}/confusion_matrix.npy", metrics['confusion_matrix'])
        
        # Save metrics summary
        summary = {
            'accuracy': metrics['accuracy'],
            'micro_f1': metrics['micro_f1'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'test_loss': metrics['test_loss'],
            'modality_importance': metrics['modality_importance']
        }
        
        import json
        with open(f"{save_path}/metrics_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Results saved to {save_path}")
    
    return metrics


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores
        
    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    
    tpr = tps / tps[-1]
    fpr = fps / fps[-1]
    
    return fpr, tpr, y_score[threshold_idxs]


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate area under curve using trapezoidal rule.
    
    Args:
        x: X values
        y: Y values
        
    Returns:
        Area under curve
    """
    return np.trapz(y, x)