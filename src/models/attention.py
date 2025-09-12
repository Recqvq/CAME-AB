import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CrossAttention(nn.Module):
    """
    Cross-attention module for attending between different modalities.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                need_weights: bool = False) -> torch.Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor
            key_value: Key and value tensor
            need_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        attn_output, attn_weights = self.multihead_attn(
            query=query, key=key_value, value=key_value, need_weights=True
        )
        attn_output = self.dropout(attn_output)
        attn_output = self.norm(query + attn_output)
        
        if need_weights:
            return attn_output, attn_weights
        return attn_output


class GatedCrossModalAttention(nn.Module):
    """
    Gated cross-modal attention for weighted fusion of different modalities.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(GatedCrossModalAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(embed_dim, num_heads, dropout) for _ in range(4)
        ])
        self.gate_norm = nn.LayerNorm(embed_dim)
        self.gate_linear = nn.Linear(embed_dim, 5)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, modal_idx: int, 
                need_weights: bool = False) -> torch.Tensor:
        """
        Forward pass of gated cross-modal attention.
        
        Args:
            x: Input tensor containing all modalities
            modal_idx: Index of the current modality
            need_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size = x.size(0)
        query = x[:, modal_idx:modal_idx+1, :]
        
        # Self-attention on current modality
        self_output, self_weights = self.self_attn(
            query, query, query, need_weights=True
        )
        
        # Cross-attention with other modalities
        other_modal_indices = [i for i in range(5) if i != modal_idx]
        cross_outputs = []
        cross_weights = []
        
        for i, other_idx in enumerate(other_modal_indices):
            other_modal = x[:, other_idx:other_idx+1, :]
            cross_out, cross_weight = self.cross_attn_layers[i](
                query, other_modal, need_weights=True
            )
            cross_outputs.append(cross_out)
            cross_weights.append(cross_weight)
        
        # Gated fusion
        gate_input = self.gate_norm(query)
        gates = F.softmax(self.gate_linear(gate_input), dim=-1)
        
        combined = gates[:, :, 0:1] * self_output
        for i, cross_out in enumerate(cross_outputs):
            combined += gates[:, :, i+1:i+2] * cross_out
        
        output = self.final_norm(query + self.dropout(combined))
        
        if need_weights:
            return output, self_weights, cross_weights, gates
        return output


class ShortcutAwareSelfAttention(nn.Module):
    """
    Self-attention module with shortcut connections and modality importance weighting.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(ShortcutAwareSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.modality_weights = nn.Parameter(torch.ones(5) * 0.2)
        
        self.importance_proj = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, need_weights: bool = False) -> torch.Tensor:
        """
        Forward pass of shortcut-aware self-attention.
        
        Args:
            x: Input tensor
            need_weights: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Calculate importance scores
        importance = self.importance_proj(x).squeeze(-1)
        importance = F.softmax(importance, dim=1)
        
        # Multi-head self-attention
        attn_output, attn_weights = self.multihead_attn(x, x, x, need_weights=True)
        
        # Apply modality weights
        modality_weights = F.softmax(self.modality_weights, dim=0).view(1, seq_len, 1)
        weighted_output = attn_output * modality_weights
        
        # Apply importance weights and residual connection
        importance = importance.unsqueeze(-1)
        residual_weight = torch.sigmoid(self.residual_weight)
        final_output = residual_weight * weighted_output * importance + (1 - residual_weight) * x
        final_output = self.norm(final_output)
        
        if need_weights:
            return final_output, attn_weights, importance.squeeze(-1), F.softmax(self.modality_weights, dim=0)
        return final_output