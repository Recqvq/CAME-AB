import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .attention import GatedCrossModalAttention, ShortcutAwareSelfAttention


class EnhancedTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with multi-attention mechanisms.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        super(EnhancedTransformerLayer, self).__init__()
        
        self.self_attn = ShortcutAwareSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.gated_cross_attns = nn.ModuleList([
            GatedCrossModalAttention(embed_dim, num_heads, dropout) for _ in range(5)
        ])
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Linear(embed_dim * 2, 1)
        
    def forward(self, src: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        """
        Forward pass of enhanced transformer layer.
        
        Args:
            src: Source tensor
            need_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, embed_dim = src.size()
        
        # Global self-attention
        if need_weights:
            global_attn, self_attn_weights, importance, modality_weights = self.self_attn(src, need_weights=True)
        else:
            global_attn = self.self_attn(src)
        
        # Cross-modal attention for each modality
        modal_outputs = []
        all_cross_weights = []
        all_gate_weights = []
        
        for i in range(seq_len):
            if need_weights:
                modal_out, _, cross_weights, gate_weights = self.gated_cross_attns[i](src, i, need_weights=True)
                all_cross_weights.append(cross_weights)
                all_gate_weights.append(gate_weights)
            else:
                modal_out = self.gated_cross_attns[i](src, i)
            modal_outputs.append(modal_out)
        
        modal_combined = torch.cat(modal_outputs, dim=1)
        
        # Gated fusion of global and modal attention
        gate_input = torch.cat([global_attn, modal_combined], dim=-1)
        gates = torch.sigmoid(self.fusion_gate(gate_input))
        combined = gates * global_attn + (1 - gates) * modal_combined
        
        # Add & Norm
        src = src + self.dropout(combined)
        src = self.norm1(src)
        
        # Feed-forward
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        
        if need_weights:
            return src, self_attn_weights, all_cross_weights, all_gate_weights, gates
        return src


class EnhancedMultiModalTransformer(nn.Module):
    """
    Full enhanced multi-modal transformer model.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, 
                 num_classes: int, dropout: float = 0.1):
        super(EnhancedMultiModalTransformer, self).__init__()
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(5)
        ])
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5, embed_dim))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(embed_dim, num_heads, embed_dim*4, dropout)
            for _ in range(num_layers)
        ])
        
        # Modality-specific projection heads
        self.modality_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(5)
        ])
        
        # Global projection head
        self.global_proj_head = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # Auxiliary classifiers for each modality
        self.modality_classifiers = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(5)
        ])
        
        # Modality attention for importance weighting
        self.modality_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor, return_features: bool = False, 
                return_attentions: bool = False) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, num_modalities, feature_dim)
            return_features: Whether to return intermediate features
            return_attentions: Whether to return attention weights
            
        Returns:
            Model outputs including predictions and optional features/attentions
        """
        batch_size = x.size(0)
        
        # Project each modality
        modal_features = []
        for i in range(5):
            modal_features.append(self.modality_projections[i](x[:, i, :]))
        x = torch.stack(modal_features, dim=1)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer layers
        attn_weights_list = []
        cross_attn_weights_list = []
        gate_weights_list = []
        fusion_gate_list = []
        
        for layer in self.transformer_layers:
            if return_attentions:
                x, attn_weights, cross_weights, gate_weights, fusion_gates = layer(x, need_weights=True)
                attn_weights_list.append(attn_weights)
                cross_attn_weights_list.append(cross_weights)
                gate_weights_list.append(gate_weights)
                fusion_gate_list.append(fusion_gates)
            else:
                x = layer(x)
        
        # Process modality outputs
        modal_outputs = []
        modal_projections = []
        modal_classifications = []
        
        for i in range(5):
            modal_feat = x[:, i, :]
            modal_outputs.append(modal_feat)
            modal_projections.append(self.modality_proj_heads[i](modal_feat))
            modal_classifications.append(self.modality_classifiers[i](modal_feat))
        
        # Calculate modality importance weights
        attn_scores = []
        for i in range(5):
            attn_scores.append(self.modality_attention(x[:, i, :]))
        attn_scores = torch.cat(attn_scores, dim=1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        
        # Weighted features
        weighted_features = []
        for i in range(5):
            weighted_features.append(x[:, i, :] * attn_weights[:, i])
        
        # Global projection and classification
        x_flat = x.reshape(batch_size, -1)
        global_projection = self.global_proj_head(x_flat)
        output = self.classifier(x_flat)
        
        if return_features:
            if return_attentions:
                return (output, modal_classifications, modal_projections, global_projection, 
                       attn_weights, attn_weights_list, cross_attn_weights_list, 
                       gate_weights_list, fusion_gate_list)
            return output, modal_classifications, modal_projections, global_projection, attn_weights
        
        if return_attentions:
            return (output, modal_classifications, attn_weights_list, 
                   cross_attn_weights_list, gate_weights_list, fusion_gate_list)
        
        return output, modal_classifications, attn_weights