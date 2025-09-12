import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WeightedContrastiveLoss(nn.Module):
    """
    Weighted contrastive loss for representation learning.
    
    Args:
        temperature: Temperature parameter for scaling
        weights: Optional class weights for imbalanced datasets
    """
    def __init__(self, temperature: float = 0.07, weights: Optional[torch.Tensor] = None):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.weights = weights
        
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted contrastive loss.
        
        Args:
            features: Feature representations
            labels: Ground truth labels
            
        Returns:
            Contrastive loss value
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create masks for positive and negative pairs
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        mask_self = torch.eye(mask_pos.size(0), device=mask_pos.device)
        mask_pos = mask_pos - mask_self
        
        # Numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - logits_max.detach()
        
        # Calculate loss
        exp_sim = torch.exp(similarity_matrix)
        mask_neg = 1.0 - mask_pos
        pos_sum = torch.sum(mask_pos * exp_sim, dim=1)
        all_sum = torch.sum(mask_neg * exp_sim, dim=1)
        
        loss = -torch.log((pos_sum + 1e-6) / (pos_sum + all_sum + 1e-6))
        
        # Apply weights if provided
        if self.weights is not None:
            weights = self.weights[labels.view(-1)]
            loss = loss * weights
        
        # Average over positive pairs
        num_pos = torch.sum(mask_pos, dim=1)
        loss = torch.sum(loss) / (torch.sum(num_pos) + 1e-6)
        
        return loss


class EnhancedMultiTaskLoss(nn.Module):
    """
    Multi-task loss combining cross-entropy, contrastive, and auxiliary losses.
    
    Args:
        num_classes: Number of output classes
        device: Device to run on
        class_weights: Optional class weights
        temperature: Temperature for contrastive loss
        contrastive_weight: Weight for contrastive loss
        aux_weight: Weight for auxiliary loss
    """
    def __init__(self, num_classes: int, device: torch.device, 
                 class_weights: Optional[torch.Tensor] = None,
                 temperature: float = 0.07,
                 contrastive_weight: float = 0.2,
                 aux_weight: float = 0.3):
        super(EnhancedMultiTaskLoss, self).__init__()
        
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.contrastive_loss = WeightedContrastiveLoss(temperature=temperature, weights=class_weights)
        self.contrastive_weight = contrastive_weight
        self.aux_weight = aux_weight
        self.num_classes = num_classes
        self.device = device
        
    def forward(self, main_output: torch.Tensor, 
                modal_outputs: list,
                modal_projections: list,
                global_projection: torch.Tensor,
                targets: torch.Tensor,
                epoch: int = 0,
                max_epochs: int = 50) -> Tuple[torch.Tensor, ...]:
        """
        Calculate combined multi-task loss.
        
        Args:
            main_output: Main model output
            modal_outputs: List of modality-specific outputs
            modal_projections: List of modality-specific projections
            global_projection: Global feature projection
            targets: Ground truth labels
            epoch: Current epoch number
            max_epochs: Total number of epochs
            
        Returns:
            Tuple of (total_loss, main_loss, aux_loss, contrastive_loss)
        """
        # Main classification loss
        main_loss = self.ce_loss(main_output, targets)
        
        # Auxiliary losses for each modality
        aux_losses = []
        for modal_output in modal_outputs:
            aux_losses.append(self.ce_loss(modal_output, targets))
        aux_loss = sum(aux_losses) / len(aux_losses)
        
        # Contrastive losses for modality projections
        modal_contrastive_losses = []
        for modal_proj in modal_projections:
            modal_contrastive_losses.append(self.contrastive_loss(modal_proj, targets))
        modal_contrastive_loss = sum(modal_contrastive_losses) / len(modal_contrastive_losses)
        
        # Global contrastive loss
        global_contrastive_loss = self.contrastive_loss(global_projection, targets)
        
        # Cross-modal consistency loss
        cross_modal_loss = 0.0
        for i in range(len(modal_projections)):
            for j in range(i+1, len(modal_projections)):
                sim = F.cosine_similarity(modal_projections[i], modal_projections[j], dim=1)
                cross_modal_loss += (1.0 - sim).mean()
        
        if len(modal_projections) > 1:
            cross_modal_loss /= (len(modal_projections) * (len(modal_projections) - 1) / 2)
        
        # Dynamic weighting based on epoch
        contrastive_weight = self.contrastive_weight * (1.0 - 0.5 * epoch / max_epochs)
        aux_weight = self.aux_weight * (1.0 - 0.7 * epoch / max_epochs)
        
        # Combine contrastive losses
        contrastive_total = (modal_contrastive_loss + global_contrastive_loss + cross_modal_loss) / 3
        
        # Total loss
        total_loss = main_loss + aux_weight * aux_loss + contrastive_weight * contrastive_total
        
        return total_loss, main_loss, aux_loss, contrastive_total