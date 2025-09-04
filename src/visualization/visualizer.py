import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training curves from training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    epochs = len(history['train_loss'])
    
    fig = plt.figure(figsize=(20, 15))
    
    # Loss curves
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(3, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # F1 Score curves
    plt.subplot(3, 2, 3)
    plt.plot(history['test_micro_f1'], label='Micro F1')
    plt.plot(history['test_macro_f1'], label='Macro F1')
    plt.plot(history['test_weighted_f1'], label='Weighted F1')
    plt.title('F1 Score Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Component losses
    plt.subplot(3, 2, 4)
    plt.plot(history['main_loss'], label='Main Loss')
    plt.plot(history['aux_loss'], label='Auxiliary Loss')
    plt.plot(history['contrastive_loss'], label='Contrastive Loss')
    plt.title('Component Losses', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Learning rate
    plt.subplot(3, 2, 5)
    plt.plot(history['lr'])
    plt.title('Learning Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    latest_metrics = {
        'Train Loss': history['train_loss'][-1],
        'Test Loss': history['test_loss'][-1],
        'Train Accuracy': history['train_acc'][-1],
        'Test Accuracy': history['test_acc'][-1],
        'Micro F1': history['test_micro_f1'][-1],
        'Macro F1': history['test_macro_f1'][-1],
        'Weighted F1': history['test_weighted_f1'][-1]
    }
    
    best_epoch = np.argmax(history['test_micro_f1'])
    best_metrics = {
        'Best Epoch': best_epoch + 1,
        'Best Micro F1': history['test_micro_f1'][best_epoch],
        'Best Macro F1': history['test_macro_f1'][best_epoch],
        'Best Weighted F1': history['test_weighted_f1'][best_epoch],
        'Best Test Accuracy': history['test_acc'][best_epoch]
    }
    
    summary_text = "Training Summary (Latest):\n\n"
    for metric, value in latest_metrics.items():
        summary_text += f"{metric}: {value:.4f}\n"
    summary_text += "\nBest Performance:\n\n"
    for metric, value in best_metrics.items():
        if metric == 'Best Epoch':
            summary_text += f"{metric}: {value}\n"
        else:
            summary_text += f"{metric}: {value:.4f}\n"
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         num_classes: int, save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    cm = confusion_matrix(y_true, y_pred)
    
    if num_classes > 20:
        # Use log scale for large number of classes
        eps = 1e-8
        log_cm = np.log(cm + eps)
        sns.heatmap(log_cm, cmap='viridis', square=True)
        plt.title('Log-scale Confusion Matrix', fontsize=16)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True)
        plt.title('Confusion Matrix', fontsize=16)
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, 
                   num_classes: int, save_path: Optional[str] = None,
                   max_display_classes: int = 10):
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
        num_classes: Number of classes
        save_path: Optional path to save the plot
        max_display_classes: Maximum number of classes to display
    """
    from src.evaluation.evaluator import roc_curve, auc
    
    plt.figure(figsize=(12, 10))
    
    # Sample classes if too many
    if num_classes > max_display_classes:
        display_classes = max_display_classes
        class_indices = np.random.choice(num_classes, display_classes, replace=False)
    else:
        display_classes = num_classes
        class_indices = range(num_classes)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, display_classes))
    
    for i, class_idx in enumerate(class_indices):
        binary_targets = (y_true == class_idx).astype(int)
        class_probs = y_probs[:, class_idx]
        
        if len(np.unique(binary_targets)) > 1:  # Both classes exist
            fpr, tpr, _ = roc_curve(binary_targets, class_probs)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'Class {class_idx} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    
    plt.show()
    plt.close()


def visualize_features(model: nn.Module, data_loader: DataLoader, 
                      device: torch.device, num_samples: int = 500,
                      epoch: Optional[int] = None, name: str = "",
                      save_path: Optional[str] = None):
    """
    Visualize features using t-SNE.
    
    Args:
        model: Model to extract features from
        data_loader: Data loader
        device: Device to run on
        num_samples: Number of samples to visualize
        epoch: Current epoch (for title)
        name: Name for the visualization
        save_path: Optional path to save the plot
        
    Returns:
        Tuple of (embedding, labels)
    """
    model.eval()
    features = []
    labels = []
    count = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            if count >= num_samples:
                break
            
            batch_size = inputs.size(0)
            remaining = min(batch_size, num_samples - count)
            
            inputs = inputs[:remaining].to(device)
            targets = targets[:remaining].to(device)
            
            # Extract features
            _, _, modal_projections, global_projection, _ = model(inputs, return_features=True)
            
            features.append(global_projection.cpu().numpy())
            labels.append(targets.cpu().numpy())
            count += remaining
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    # Apply t-SNE
    logger.info("Applying t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1], c=[colors[i]], 
                   label=f'Class {label}', alpha=0.7, s=50, 
                   edgecolors='k', linewidths=0.5)
    
    title = f'{name} Feature Space (t-SNE)'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    
    if len(unique_labels) > 20:
        plt.legend(fontsize=8, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature visualization saved to {save_path}")
    
    plt.show()
    plt.close()
    
    return embedding, labels


def plot_modality_importance(attention_weights: Dict[str, float], 
                            save_path: Optional[str] = None):
    """
    Plot modality importance weights.
    
    Args:
        attention_weights: Dictionary of modality names and their importance weights
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    modality_names = list(attention_weights.keys())
    weights = list(attention_weights.values())
    
    plt.bar(modality_names, weights, color='skyblue')
    plt.title('Average Modality Importance Weights', fontsize=16)
    plt.ylabel('Importance Weight', fontsize=14)
    plt.xticks(fontsize=12)
    plt.ylim(0, max(weights) * 1.2)
    
    # Add value labels on bars
    for i, v in enumerate(weights):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Modality importance plot saved to {save_path}")
    
    plt.show()
    plt.close()