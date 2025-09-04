import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import logging
import time
import os
import random
from typing import Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for model training and validation.
    
    Args:
        model: Model to train
        device: Device to run on
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
    """
    def __init__(self, model: nn.Module, device: torch.device,
                 criterion: nn.Module, optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 scaler: Optional[GradScaler] = None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler or GradScaler()
        
    def train_epoch(self, train_loader: DataLoader, epoch: int, 
                   num_epochs: int, mixup_prob: float = 0.3) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            num_epochs: Total number of epochs
            mixup_prob: Probability of applying mixup
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_contrastive_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Apply mixup augmentation
            if random.random() < mixup_prob:
                mixed_inputs, labels_a, labels_b, lam = self._mixup(inputs, labels)
                inputs = mixed_inputs
                mixup_applied = True
            else:
                mixup_applied = False
            
            self.optimizer.zero_grad()
            
            with autocast():
                outputs, modal_outputs, modal_projections, global_projection, _ = self.model(
                    inputs, return_features=True
                )
                
                if mixup_applied:
                    loss_a, main_loss_a, aux_loss_a, contrastive_loss_a = self.criterion(
                        outputs, modal_outputs, modal_projections, global_projection,
                        labels_a, epoch, num_epochs
                    )
                    loss_b, main_loss_b, aux_loss_b, contrastive_loss_b = self.criterion(
                        outputs, modal_outputs, modal_projections, global_projection,
                        labels_b, epoch, num_epochs
                    )
                    loss = lam * loss_a + (1 - lam) * loss_b
                    main_loss = lam * main_loss_a + (1 - lam) * main_loss_b
                    aux_loss = lam * aux_loss_a + (1 - lam) * aux_loss_b
                    contrastive_loss = lam * contrastive_loss_a + (1 - lam) * contrastive_loss_b
                else:
                    loss, main_loss, aux_loss, contrastive_loss = self.criterion(
                        outputs, modal_outputs, modal_projections, global_projection,
                        labels, epoch, num_epochs
                    )
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            epoch_main_loss += main_loss.item() * inputs.size(0)
            epoch_aux_loss += aux_loss.item() * inputs.size(0)
            epoch_contrastive_loss += contrastive_loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            if mixup_applied:
                train_total += inputs.size(0)
                train_correct += (lam * (predicted == labels_a).sum().item() + 
                                 (1 - lam) * (predicted == labels_b).sum().item())
            else:
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
        if self.scheduler:
            self.scheduler.step()
        
        return {
            'loss': train_loss / len(train_loader.dataset),
            'accuracy': train_correct / train_total,
            'main_loss': epoch_main_loss / len(train_loader.dataset),
            'aux_loss': epoch_aux_loss / len(train_loader.dataset),
            'contrastive_loss': epoch_contrastive_loss / len(train_loader.dataset)
        }
    
    def validate(self, val_loader: DataLoader, epoch: int, 
                num_epochs: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            num_epochs: Total number of epochs
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs, modal_outputs, modal_projections, global_projection, _ = self.model(
                    inputs, return_features=True
                )
                
                loss, _, _, _ = self.criterion(
                    outputs, modal_outputs, modal_projections, global_projection,
                    labels, epoch, num_epochs
                )
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_micro_f1 = f1_score(val_targets, val_preds, average='micro')
        val_macro_f1 = f1_score(val_targets, val_preds, average='macro')
        val_weighted_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'micro_f1': val_micro_f1,
            'macro_f1': val_macro_f1,
            'weighted_f1': val_weighted_f1
        }
    
    def _mixup(self, features: torch.Tensor, labels: torch.Tensor, 
               alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation."""
        batch_size = features.shape[0]
        indices = torch.randperm(batch_size)
        lam = np.random.beta(alpha, alpha)
        mixed_features = lam * features + (1 - lam) * features[indices]
        return mixed_features, labels, labels[indices], lam


def train_enhanced_model(model: nn.Module,
                        train_loader: DataLoader,
                        test_loader: DataLoader,
                        criterion: nn.Module,
                        optimizer: optim.Optimizer,
                        scheduler: optim.lr_scheduler._LRScheduler,
                        device: torch.device,
                        num_epochs: int = 50,
                        patience: int = 10,
                        save_path: str = './models/best_model.pth',
                        recreate_sampler_interval: int = 5) -> Tuple[Dict, float]:
    """
    Train the enhanced multi-modal model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to run on
        num_epochs: Number of epochs
        patience: Early stopping patience
        save_path: Path to save best model
        recreate_sampler_interval: Interval to recreate weighted sampler
        
    Returns:
        Training history and best F1 score
    """
    logger.info("Starting enhanced model training...")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'test_micro_f1': [], 'test_macro_f1': [], 'test_weighted_f1': [],
        'main_loss': [], 'aux_loss': [], 'contrastive_loss': [],
        'lr': []
    }
    
    best_micro_f1 = 0.0
    patience_counter = 0
    trainer = Trainer(model, device, criterion, optimizer, scheduler)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch, num_epochs)
        
        # Validate
        test_metrics = trainer.validate(test_loader, epoch, num_epochs)
        
        # Save best model
        if test_metrics['micro_f1'] > best_micro_f1:
            best_micro_f1 = test_metrics['micro_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_micro_f1
            }, save_path)
            logger.info(f"Saved new best model with micro F1 score: {best_micro_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['test_loss'].append(test_metrics['loss'])
        history['test_acc'].append(test_metrics['accuracy'])
        history['test_micro_f1'].append(test_metrics['micro_f1'])
        history['test_macro_f1'].append(test_metrics['macro_f1'])
        history['test_weighted_f1'].append(test_metrics['weighted_f1'])
        history['main_loss'].append(train_metrics['main_loss'])
        history['aux_loss'].append(train_metrics['aux_loss'])
        history['contrastive_loss'].append(train_metrics['contrastive_loss'])
        
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Log metrics
        time_elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {time_elapsed:.2f}s")
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test F1 - Micro: {test_metrics['micro_f1']:.4f}, "
                   f"Macro: {test_metrics['macro_f1']:.4f}, "
                   f"Weighted: {test_metrics['weighted_f1']:.4f}")
    
    return history, best_micro_f1