#!/usr/bin/env python3
"""
Multi-Modal Transformer for Classification
Main training and evaluation script.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import (
    load_config, set_seed, setup_logging, 
    get_device, count_parameters, load_checkpoint
)
from src.data import (
    load_features, normalize_features, create_modality_tensors,
    MultiModalDataset, create_weighted_sampler
)
from src.models import EnhancedMultiModalTransformer, EnhancedMultiTaskLoss
from src.training import train_enhanced_model
from src.evaluation import evaluate_model
from src.visualization import (
    plot_training_curves, plot_confusion_matrix,
    plot_modality_importance, visualize_features
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-Modal Transformer Training')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation or resume training')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def prepare_data(config):
    """Load and prepare data based on configuration."""
    logging.info("Loading features...")
    
    # Load all features
    features = load_features(config)
    
    # Merge train and validation for training
    import numpy as np
    
    esmc_train_val = np.concatenate([features['esmc_train'], features['esmc_val']], axis=0)
    esmc_train_val_labels = np.concatenate([features['esmc_train_labels'], features['esmc_val_labels']], axis=0)
    
    esmcStructure_train_val = np.concatenate([features['esmcStructure_train'], features['esmcStructure_val']], axis=0)
    onehot_train_val = np.concatenate([features['onehot_train'], features['onehot_val']], axis=0)
    blosum_train_val = np.concatenate([features['blosum_train'], features['blosum_val']], axis=0)
    
    # Split GCN features
    esm_train_ratio = len(features['esmc_train']) / (len(features['esmc_train']) + len(features['esmc_val']))
    gcn_train_size = int(len(features['gcn_train_val']) * esm_train_ratio)
    shuffled_indices = np.random.permutation(len(features['gcn_train_val']))
    gcn_train = features['gcn_train_val'][shuffled_indices[:gcn_train_size]]
    gcn_val = features['gcn_train_val'][shuffled_indices[gcn_train_size:]]
    gcn_train_val = np.concatenate([gcn_train, gcn_val], axis=0)
    
    # Create modality tensors
    train_val_modality_tensors = create_modality_tensors(
        normalize_features(esmc_train_val),
        normalize_features(esmcStructure_train_val),
        normalize_features(onehot_train_val),
        normalize_features(blosum_train_val),
        normalize_features(gcn_train_val)
    )
    
    test_modality_tensors = create_modality_tensors(
        normalize_features(features['esmc_test']),
        normalize_features(features['esmcStructure_test']),
        normalize_features(features['onehot_test']),
        normalize_features(features['blosum_test']),
        normalize_features(features['gcn_test'])
    )
    
    # Get labels
    y_train_val = esmc_train_val_labels
    y_test = features['esmc_test_labels']
    
    # Create datasets
    train_val_dataset = MultiModalDataset(
        train_val_modality_tensors, y_train_val,
        augment=config['augmentation']['enabled'],
        augment_prob=config['augmentation']['augment_prob']
    )
    
    test_dataset = MultiModalDataset(test_modality_tensors, y_test, augment=False)
    
    # Create data loaders
    weighted_sampler = create_weighted_sampler(y_train_val)
    
    train_val_loader = DataLoader(
        train_val_dataset,
        batch_size=config['training']['batch_size'],
        sampler=weighted_sampler,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Get number of classes
    num_classes = len(np.unique(np.concatenate([y_train_val, y_test])))
    
    logging.info(f"Data loaded - Train+Val: {len(train_val_dataset)}, Test: {len(test_dataset)}")
    logging.info(f"Number of classes: {num_classes}")
    
    return train_val_loader, test_loader, num_classes


def create_model(config, num_classes, device):
    """Create model, loss, optimizer, and scheduler."""
    
    # Create model
    model = EnhancedMultiModalTransformer(
        embed_dim=config['model']['embed_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        num_classes=num_classes,
        dropout=config['model']['dropout']
    ).to(device)
    
    logging.info(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Create loss
    criterion = EnhancedMultiTaskLoss(
        num_classes=num_classes,
        device=device,
        temperature=config['loss']['temperature'],
        contrastive_weight=config['loss']['contrastive_weight'],
        aux_weight=config['loss']['aux_weight']
    )
    
    # Create optimizer
    if config['training']['optimizer']['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['training']['optimizer']['betas']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']['type']}")
    
    # Create scheduler
    if config['training']['scheduler']['type'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training']['scheduler']['T_0'],
            T_mult=config['training']['scheduler']['T_mult'],
            eta_min=config['training']['scheduler']['eta_min']
        )
    else:
        scheduler = None
    
    return model, criterion, optimizer, scheduler


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.seed:
        config['seed'] = args.seed
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    
    # Setup
    set_seed(config['seed'])
    
    # Create output directories
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['visualization_dir'], exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(config['output']['save_dir'], config['output']['log_file'])
    setup_logging(log_file=log_file)
    
    logging.info(f"Configuration loaded from {args.config}")
    
    # Get device
    if args.device == 'auto':
        device = get_device(force_cpu=not config['hardware']['use_gpu'])
    else:
        device = torch.device(args.device)
    
    # Prepare data
    train_loader, test_loader, num_classes = prepare_data(config)
    
    # Create model
    model, criterion, optimizer, scheduler = create_model(config, num_classes, device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    
    # Training mode
    if args.mode in ['train', 'both']:
        logging.info("Starting training...")
        
        # Train model
        history, best_f1 = train_enhanced_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=config['training']['num_epochs'],
            patience=config['training']['patience'],
            save_path=os.path.join(config['output']['model_dir'], 'best_model.pth')
        )
        
        # Plot training curves
        plot_training_curves(
            history,
            save_path=os.path.join(config['output']['visualization_dir'], 'training_curves.png')
        )
        
        logging.info(f"Training completed. Best F1: {best_f1:.4f}")
    
    # Evaluation mode
    if args.mode in ['eval', 'both']:
        logging.info("Starting evaluation...")
        
        # Load best model for evaluation
        if args.mode == 'both':
            checkpoint_path = os.path.join(config['output']['model_dir'], 'best_model.pth')
        else:
            checkpoint_path = args.checkpoint or os.path.join(config['output']['model_dir'], 'best_model.pth')
        
        checkpoint = load_checkpoint(checkpoint_path, model, device=device)
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            save_path=config['output']['save_dir']
        )
        
        # Visualizations
        # Plot confusion matrix
        if 'confusion_matrix' in results:
            plot_confusion_matrix(
                results['confusion_matrix'],
                num_classes=num_classes,
                save_path=os.path.join(config['output']['visualization_dir'], 'confusion_matrix.png')
            )
        
        # Plot modality importance
        if 'modality_importance' in results:
            plot_modality_importance(
                results['modality_importance'],
                save_path=os.path.join(config['output']['visualization_dir'], 'modality_importance.png')
            )
        
        # Feature visualization
        try:
            visualize_features(
                model=model,
                data_loader=test_loader,
                device=device,
                num_samples=500,
                name="Test",
                save_path=os.path.join(config['output']['visualization_dir'], 'feature_space.png')
            )
        except Exception as e:
            logging.warning(f"Feature visualization failed: {e}")
        
        logging.info("Evaluation completed")
        logging.info(f"Final Results:")
        logging.info(f"  Accuracy: {results['accuracy']:.4f}")
        logging.info(f"  Micro F1: {results['micro_f1']:.4f}")
        logging.info(f"  Macro F1: {results['macro_f1']:.4f}")
        logging.info(f"  Weighted F1: {results['weighted_f1']:.4f}")


if __name__ == "__main__":
    main()