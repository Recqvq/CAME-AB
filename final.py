# Import necessary libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
import random
from itertools import cycle

warnings.filterwarnings('ignore')

# Setup logging with more detailed formatting
logging.basicConfig(filename='enhanced_transformer_training.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed()

# Set device with better error handling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

# Function to create directories for saving results
def create_output_directories():
    directories = ['./models', './results', './visualizations']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

create_output_directories()

# Load features from each model - UPDATED PATHS
try:
    # ESMC features
    esmc_train_features = np.load('/home/wrj/Rec/A/Esmc/esmc_train_features.npy')
    esmc_train_labels = np.load('/home/wrj/Rec/A/Esmc/esmc_train_labels.npy')
    esmc_val_features = np.load('/home/wrj/Rec/A/Esmc/esmc_val_features.npy')
    esmc_val_labels = np.load('/home/wrj/Rec/A/Esmc/esmc_val_labels.npy')
    esmc_test_features = np.load('/home/wrj/Rec/A/Esmc/esmc_test_features.npy')
    esmc_test_labels = np.load('/home/wrj/Rec/A/Esmc/esmc_test_labels.npy')
    
    # ESMCStructure features
    esmcStructure_train_features = np.load('/home/wrj/Rec/A/EsmcStructure/esmc_multifeature_train_features.npy')
    esmcStructure_train_labels = np.load('/home/wrj/Rec/A/EsmcStructure/esmc_multifeature_train_labels.npy')
    esmcStructure_val_features = np.load('/home/wrj/Rec/A/EsmcStructure/esmc_multifeature_val_features.npy')
    esmcStructure_val_labels = np.load('/home/wrj/Rec/A/EsmcStructure/esmc_multifeature_val_labels.npy')
    esmcStructure_test_features = np.load('/home/wrj/Rec/A/EsmcStructure/esmc_multifeature_test_features.npy')
    esmcStructure_test_labels = np.load('/home/wrj/Rec/A/EsmcStructure/esmc_multifeature_test_labels.npy')
    
    # OneHot features
    onehot_train_features = np.load('/home/wrj/Rec/A/OneHot/onehot_train_features.npy')
    onehot_train_labels = np.load('/home/wrj/Rec/A/OneHot/onehot_train_labels.npy')
    onehot_val_features = np.load('/home/wrj/Rec/A/OneHot/onehot_val_features.npy')
    onehot_val_labels = np.load('/home/wrj/Rec/A/OneHot/onehot_val_labels.npy')
    onehot_test_features = np.load('/home/wrj/Rec/A/OneHot/onehot_test_features.npy')
    onehot_test_labels = np.load('/home/wrj/Rec/A/OneHot/onehot_test_labels.npy')
    
    # BLOSUM features
    blosum_train_features = np.load('/home/wrj/Rec/A/BLOSUM/blosum_train_features.npy')
    blosum_train_labels = np.load('/home/wrj/Rec/A/BLOSUM/blosum_train_labels.npy')
    blosum_val_features = np.load('/home/wrj/Rec/A/BLOSUM/blosum_val_features.npy')
    blosum_val_labels = np.load('/home/wrj/Rec/A/BLOSUM/blosum_val_labels.npy')
    blosum_test_features = np.load('/home/wrj/Rec/A/BLOSUM/blosum_test_features.npy')
    blosum_test_labels = np.load('/home/wrj/Rec/A/BLOSUM/blosum_test_labels.npy')
    
    # GCN features - train_val combined features
    gcn_train_val_features = np.load('/home/wrj/Rec/A/GCN/gcn_train_val_features.npy')
    gcn_train_val_labels = np.load('/home/wrj/Rec/A/GCN/gcn_train_val_labels.npy')
    gcn_test_features = np.load('/home/wrj/Rec/A/GCN/gcn_test_features.npy')
    gcn_test_labels = np.load('/home/wrj/Rec/A/GCN/gcn_test_labels.npy')
    
    # Split GCN features into train and val to match other datasets
    esm_train_ratio = len(esmc_train_features) / (len(esmc_train_features) + len(esmc_val_features))
    gcn_train_size = int(len(gcn_train_val_features) * esm_train_ratio)
    shuffled_indices = np.random.permutation(len(gcn_train_val_features))
    gcn_train_features = gcn_train_val_features[shuffled_indices[:gcn_train_size]]
    gcn_train_labels = gcn_train_val_labels[shuffled_indices[:gcn_train_size]]
    gcn_val_features = gcn_train_val_features[shuffled_indices[gcn_train_size:]]
    gcn_val_labels = gcn_train_val_labels[shuffled_indices[gcn_train_size:]]
    
    logging.info(f"Loaded ESMC train features: {esmc_train_features.shape}, val: {esmc_val_features.shape}, test: {esmc_test_features.shape}")
    logging.info(f"Loaded ESMCStructure train features: {esmcStructure_train_features.shape}, val: {esmcStructure_val_features.shape}, test: {esmcStructure_test_features.shape}")
    logging.info(f"Loaded OneHot train features: {onehot_train_features.shape}, val: {onehot_val_features.shape}, test: {onehot_test_features.shape}")
    logging.info(f"Loaded BLOSUM train features: {blosum_train_features.shape}, val: {blosum_val_features.shape}, test: {blosum_test_features.shape}")
    logging.info(f"Split GCN features into train: {gcn_train_features.shape}, val: {gcn_val_features.shape}, test: {gcn_test_features.shape}")
    
    print("Loaded all feature sets successfully")
    
except FileNotFoundError as e:
    logging.error(f"Error loading features: {e}")
    print(f"Error loading features: {e}")
    raise

# Merge train and validation data for training
esmc_train_val_features = np.concatenate([esmc_train_features, esmc_val_features], axis=0)
esmc_train_val_labels = np.concatenate([esmc_train_labels, esmc_val_labels], axis=0)

blosum_train_val_features = np.concatenate([blosum_train_features, blosum_val_features], axis=0)
blosum_train_val_labels = np.concatenate([blosum_train_labels, blosum_val_labels], axis=0)

onehot_train_val_features = np.concatenate([onehot_train_features, onehot_val_features], axis=0)
onehot_train_val_labels = np.concatenate([onehot_train_labels, onehot_val_labels], axis=0)

esmcStructure_train_val_features = np.concatenate([esmcStructure_train_features, esmcStructure_val_features], axis=0)
esmcStructure_train_val_labels = np.concatenate([esmcStructure_train_labels, esmcStructure_val_labels], axis=0)

gcn_train_val_features = np.concatenate([gcn_train_features, gcn_val_features], axis=0)
gcn_train_val_labels = np.concatenate([gcn_train_labels, gcn_val_labels], axis=0)

logging.info(f"Combined train+val features - ESMC: {esmc_train_val_features.shape}, ESMCStructure: {esmcStructure_train_val_features.shape}, "
             f"OneHot: {onehot_train_val_features.shape}, BLOSUM: {blosum_train_val_features.shape}, GCN: {gcn_train_val_features.shape}")

print(f"Combined train+val features - ESMC: {esmc_train_val_features.shape}, ESMCStructure: {esmcStructure_train_val_features.shape}, "
      f"OneHot: {onehot_train_val_features.shape}, BLOSUM: {blosum_train_val_features.shape}, GCN: {gcn_train_val_features.shape}")

# Check if features are already 256-dimensional
feature_dims = [
    esmc_train_val_features.shape[1],
    esmcStructure_train_val_features.shape[1],
    onehot_train_val_features.shape[1],
    blosum_train_val_features.shape[1],
    gcn_train_val_features.shape[1]
]

for i, modality in enumerate(['ESMC', 'ESMCStructure', 'OneHot', 'BLOSUM', 'GCN']):
    print(f"{modality} feature dimension: {feature_dims[i]}")
    if feature_dims[i] != 256:
        logging.warning(f"{modality} features are not 256-dimensional (actual: {feature_dims[i]})")

# Feature augmentation with mixup
def feature_mixup(features, labels, alpha=0.2):
    """Create mixup samples by linearly interpolating between two random examples."""
    batch_size = features.shape[0]
    indices = torch.randperm(batch_size)
    lam = np.random.beta(alpha, alpha)
    mixed_features = lam * features + (1 - lam) * features[indices]
    return mixed_features, labels, labels[indices], lam

# Normalize features
def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

# Create feature modality tensors for the transformer
def create_modality_tensors(esmc, esmcStructure, onehot, blosum, gcn):
    batch_size = esmc.shape[0]
    modality_tensors = np.zeros((batch_size, 5, 256), dtype=np.float32)
    modality_tensors[:, 0, :] = esmc
    modality_tensors[:, 1, :] = esmcStructure
    modality_tensors[:, 2, :] = onehot
    modality_tensors[:, 3, :] = blosum
    modality_tensors[:, 4, :] = gcn
    return modality_tensors

# Create modality tensors for train+val and test
train_val_modality_tensors = create_modality_tensors(
    normalize_features(esmc_train_val_features),
    normalize_features(esmcStructure_train_val_features),
    normalize_features(onehot_train_val_features),
    normalize_features(blosum_train_val_features),
    normalize_features(gcn_train_val_features)
)
test_modality_tensors = create_modality_tensors(
    normalize_features(esmc_test_features),
    normalize_features(esmcStructure_test_features),
    normalize_features(onehot_test_features),
    normalize_features(blosum_test_features),
    normalize_features(gcn_test_features)
)

print(f"Train+val modality tensors shape: {train_val_modality_tensors.shape}")
print(f"Test modality tensors shape: {test_modality_tensors.shape}")

# Get labels for training and testing
y_train_val = esmc_train_val_labels
y_test = esmc_test_labels

unique_classes = np.unique(np.concatenate([y_train_val, y_test]))
num_classes = len(unique_classes)
print(f"Number of classes: {num_classes}")
logging.info(f"Number of classes: {num_classes}")

# Custom dataset for stratified sampling and feature augmentation
class MultiModalDataset(Dataset):
    def __init__(self, features, labels, augment=False, augment_prob=0.3):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
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

# Create datasets and dataloaders
train_val_dataset = MultiModalDataset(train_val_modality_tensors, y_train_val, augment=True)
test_dataset = MultiModalDataset(test_modality_tensors, y_test, augment=False)

def create_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

weighted_sampler = create_weighted_sampler(y_train_val)
batch_size = 64
train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# Define Cross-Attention Module
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value, need_weights=False):
        attn_output, attn_weights = self.multihead_attn(query=query, key=key_value, value=key_value, need_weights=True)
        attn_output = self.dropout(attn_output)
        attn_output = self.norm(query + attn_output)
        if need_weights:
            return attn_output, attn_weights
        return attn_output

# Define Gated Cross-Modal Attention
class GatedCrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GatedCrossModalAttention, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(embed_dim, num_heads, dropout) for _ in range(4)
        ])
        self.gate_norm = nn.LayerNorm(embed_dim)
        self.gate_linear = nn.Linear(embed_dim, 5)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, modal_idx, need_weights=False):
        batch_size = x.size(0)
        query = x[:, modal_idx:modal_idx+1, :]
        self_output, self_weights = self.self_attn(query, query, query, need_weights=True)
        other_modal_indices = [i for i in range(5) if i != modal_idx]
        cross_outputs = []
        cross_weights = []
        for i, other_idx in enumerate(other_modal_indices):
            other_modal = x[:, other_idx:other_idx+1, :]
            cross_out, cross_weight = self.cross_attn_layers[i](query, other_modal, need_weights=True)
            cross_outputs.append(cross_out)
            cross_weights.append(cross_weight)
        gate_input = self.gate_norm(query)
        gates = F.softmax(self.gate_linear(gate_input), dim=-1)
        combined = gates[:, :, 0:1] * self_output
        for i, cross_out in enumerate(cross_outputs):
            combined += gates[:, :, i+1:i+2] * cross_out
        output = self.final_norm(query + self.dropout(combined))
        if need_weights:
            return output, self_weights, cross_weights, gates
        return output

# Define Shortcut-Aware Self-Attention Module
class ShortcutAwareSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(ShortcutAwareSelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
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
        
    def forward(self, x, need_weights=False):
        batch_size, seq_len, _ = x.size()
        importance = self.importance_proj(x).squeeze(-1)
        importance = F.softmax(importance, dim=1)
        attn_output, attn_weights = self.multihead_attn(x, x, x, need_weights=True)
        modality_weights = F.softmax(self.modality_weights, dim=0).view(1, seq_len, 1)
        weighted_output = attn_output * modality_weights
        importance = importance.unsqueeze(-1)
        residual_weight = torch.sigmoid(self.residual_weight)
        final_output = residual_weight * weighted_output * importance + (1 - residual_weight) * x
        final_output = self.norm(final_output)
        if need_weights:
            return final_output, attn_weights, importance.squeeze(-1), F.softmax(self.modality_weights, dim=0)
        return final_output

# Define enhanced transformer layer with multi-attention mechanisms
class EnhancedTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward=1024, dropout=0.1):
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
        
    def forward(self, src, need_weights=False):
        batch_size, seq_len, embed_dim = src.size()
        if need_weights:
            global_attn, self_attn_weights, importance, modality_weights = self.self_attn(src, need_weights=True)
        else:
            global_attn = self.self_attn(src)
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
        gate_input = torch.cat([global_attn, modal_combined], dim=-1)
        gates = torch.sigmoid(self.fusion_gate(gate_input))
        combined = gates * global_attn + (1 - gates) * modal_combined
        src = src + self.dropout(combined)
        src = self.norm1(src)
        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        if need_weights:
            return src, self_attn_weights, all_cross_weights, all_gate_weights, gates
        return src

# Define weighted contrastive loss function
class WeightedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, weights=None):
        super(WeightedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.weights = weights
        
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        mask_self = torch.eye(mask_pos.size(0), device=mask_pos.device)
        mask_pos = mask_pos - mask_self
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        similarity_matrix = similarity_matrix - logits_max.detach()
        exp_sim = torch.exp(similarity_matrix)
        mask_neg = 1.0 - mask_pos
        pos_sum = torch.sum(mask_pos * exp_sim, dim=1)
        all_sum = torch.sum(mask_neg * exp_sim, dim=1)
        loss = -torch.log((pos_sum + 1e-6) / (pos_sum + all_sum + 1e-6))
        if self.weights is not None:
            weights = self.weights[labels.view(-1)]
            loss = loss * weights
        num_pos = torch.sum(mask_pos, dim=1)
        loss = torch.sum(loss) / (torch.sum(num_pos) + 1e-6)
        return loss

# Define full enhanced transformer model
class EnhancedMultiModalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes, dropout=0.1):
        super(EnhancedMultiModalTransformer, self).__init__()
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(5)
        ])
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5, embed_dim))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerLayer(embed_dim, num_heads, embed_dim*4, dropout)
            for _ in range(num_layers)
        ])
        self.modality_proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(5)
        ])
        self.global_proj_head = nn.Sequential(
            nn.Linear(embed_dim * 5, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
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
        self.modality_classifiers = nn.ModuleList([
            nn.Linear(embed_dim, num_classes) for _ in range(5)
        ])
        self.modality_attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, x, return_features=False, return_attentions=False):
        batch_size = x.size(0)
        modal_features = []
        for i in range(5):
            modal_features.append(self.modality_projections[i](x[:, i, :]))
        x = torch.stack(modal_features, dim=1)
        x = x + self.pos_encoder
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
        modal_outputs = []
        modal_projections = []
        modal_classifications = []
        for i in range(5):
            modal_feat = x[:, i, :]
            modal_outputs.append(modal_feat)
            modal_projections.append(self.modality_proj_heads[i](modal_feat))
            modal_classifications.append(self.modality_classifiers[i](modal_feat))
        attn_scores = []
        for i in range(5):
            attn_scores.append(self.modality_attention(x[:, i, :]))
        attn_scores = torch.cat(attn_scores, dim=1)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        weighted_features = []
        for i in range(5):
            weighted_features.append(x[:, i, :] * attn_weights[:, i])
        x_flat = x.reshape(batch_size, -1)
        global_projection = self.global_proj_head(x_flat)
        output = self.classifier(x_flat)
        if return_features:
            if return_attentions:
                return output, modal_classifications, modal_projections, global_projection, attn_weights, attn_weights_list, cross_attn_weights_list, gate_weights_list, fusion_gate_list
            return output, modal_classifications, modal_projections, global_projection, attn_weights
        if return_attentions:
            return output, modal_classifications, attn_weights_list, cross_attn_weights_list, gate_weights_list, fusion_gate_list
        return output, modal_classifications, attn_weights

# Define enhanced loss function combining CE, contrastive, and auxiliary losses
class EnhancedMultiTaskLoss(nn.Module):
    def __init__(self, num_classes, device, class_weights=None, temperature=0.07, 
                 contrastive_weight=0.2, aux_weight=0.3):
        super(EnhancedMultiTaskLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        self.contrastive_loss = WeightedContrastiveLoss(temperature=temperature, weights=class_weights)
        self.contrastive_weight = contrastive_weight
        self.aux_weight = aux_weight
        self.num_classes = num_classes
        self.device = device
        
    def forward(self, main_output, modal_outputs, modal_projections, global_projection, targets, epoch=0, max_epochs=50):
        main_loss = self.ce_loss(main_output, targets)
        aux_losses = []
        for modal_output in modal_outputs:
            aux_losses.append(self.ce_loss(modal_output, targets))
        aux_loss = sum(aux_losses) / len(aux_losses)
        modal_contrastive_losses = []
        for modal_proj in modal_projections:
            modal_contrastive_losses.append(self.contrastive_loss(modal_proj, targets))
        modal_contrastive_loss = sum(modal_contrastive_losses) / len(modal_contrastive_losses)
        global_contrastive_loss = self.contrastive_loss(global_projection, targets)
        cross_modal_loss = 0.0
        for i in range(len(modal_projections)):
            for j in range(i+1, len(modal_projections)):
                sim = F.cosine_similarity(modal_projections[i], modal_projections[j], dim=1)
                cross_modal_loss += (1.0 - sim).mean()
        cross_modal_loss /= (len(modal_projections) * (len(modal_projections) - 1) / 2)
        contrastive_weight = self.contrastive_weight * (1.0 - 0.5 * epoch / max_epochs)
        aux_weight = self.aux_weight * (1.0 - 0.7 * epoch / max_epochs)
        contrastive_total = (modal_contrastive_loss + global_contrastive_loss + cross_modal_loss) / 3
        total_loss = main_loss + aux_weight * aux_loss + contrastive_weight * contrastive_total
        return total_loss, main_loss, aux_loss, contrastive_total

# Initialize model, loss, optimizer and scheduler
embed_dim = 256
num_heads = 8
num_layers = 3  
dropout = 0.3
model = EnhancedMultiModalTransformer(embed_dim, num_heads, num_layers, num_classes, dropout=dropout).to(device)
logging.info(f"Model architecture:\n{model}")

criterion = EnhancedMultiTaskLoss(num_classes, device, class_weights=None, temperature=0.07, 
                                  contrastive_weight=0.2, aux_weight=0.3)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Visualize features using TSNE instead of UMAP
def visualize_features(model, data_loader, num_samples=500, epoch=None, name=""):
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
            _, _, modal_projections, global_projection, _ = model(inputs, return_features=True)
            features.append(global_projection.cpu().numpy())
            labels.append(targets.cpu().numpy())
            count += remaining
    features = np.vstack(features)
    labels = np.concatenate(labels)
    tsne = TSNE(n_components=2, random_state=42)
    embedding = tsne.fit_transform(features)
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1], c=[colors[i]], 
                   label=f'Class {label}', alpha=0.7, s=50, edgecolors='k', linewidths=0.5)
    plt.title(f'{name} Feature Space (TSNE) - Epoch {epoch}', fontsize=16)
    plt.xlabel('TSNE Dimension 1', fontsize=14)
    plt.ylabel('TSNE Dimension 2', fontsize=14)
    if len(unique_labels) > 20:
        plt.legend(fontsize=8, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    save_path = f'./visualizations/feature_space_{name}_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Feature visualization saved to {save_path}")
    return embedding, labels

# Training function with enhanced monitoring and visualization
def train_enhanced_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                         num_epochs=50, patience=10, save_path='./models/enhanced_transformer_model.pth'):
    logging.info("Starting enhanced model training...")
    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 
        'test_micro_f1': [], 'test_macro_f1': [], 'test_weighted_f1': [],
        'main_loss': [], 'aux_loss': [], 'contrastive_loss': [],
        'lr': []
    }
    best_micro_f1 = 0.0
    patience_counter = 0
    scaler = GradScaler()
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_contrastive_loss = 0.0
        if epoch > 0 and epoch % 5 == 0:
            weighted_sampler = create_weighted_sampler(y_train_val)
            train_loader = DataLoader(train_val_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=4)
            logging.info(f"Recreated weighted sampler for epoch {epoch+1}")
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            inputs, labels = inputs.to(device), labels.to(device)
            if random.random() < 0.3:
                mixed_inputs, labels_a, labels_b, lam = feature_mixup(inputs, labels)
                inputs = mixed_inputs
                mixup_applied = True
            else:
                mixup_applied = False
            optimizer.zero_grad()
            with autocast():
                outputs, modal_outputs, modal_projections, global_projection, _ = model(inputs, return_features=True)
                if mixup_applied:
                    loss_a, main_loss_a, aux_loss_a, contrastive_loss_a = criterion(
                        outputs, modal_outputs, modal_projections, global_projection, 
                        labels_a, epoch, num_epochs
                    )
                    loss_b, main_loss_b, aux_loss_b, contrastive_loss_b = criterion(
                        outputs, modal_outputs, modal_projections, global_projection, 
                        labels_b, epoch, num_epochs
                    )
                    loss = lam * loss_a + (1 - lam) * loss_b
                    main_loss = lam * main_loss_a + (1 - lam) * main_loss_b
                    aux_loss = lam * aux_loss_a + (1 - lam) * aux_loss_b
                    contrastive_loss = lam * contrastive_loss_a + (1 - lam) * contrastive_loss_b
                else:
                    loss, main_loss, aux_loss, contrastive_loss = criterion(
                        outputs, modal_outputs, modal_projections, global_projection, 
                        labels, epoch, num_epochs
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
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
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        train_loss = train_loss / len(train_loader.dataset)
        epoch_main_loss = epoch_main_loss / len(train_loader.dataset)
        epoch_aux_loss = epoch_aux_loss / len(train_loader.dataset)
        epoch_contrastive_loss = epoch_contrastive_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        model.eval()
        test_loss = 0.0
        test_preds = []
        test_targets = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Testing)"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, modal_outputs, modal_projections, global_projection, _ = model(inputs, return_features=True)
                loss, _, _, _ = criterion(
                    outputs, modal_outputs, modal_projections, global_projection, 
                    labels, epoch, num_epochs
                )
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_targets.extend(labels.cpu().numpy())
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = accuracy_score(test_targets, test_preds)
        test_micro_f1 = f1_score(test_targets, test_preds, average='micro')
        test_macro_f1 = f1_score(test_targets, test_preds, average='macro')
        test_weighted_f1 = f1_score(test_targets, test_preds, average='weighted')
        if test_micro_f1 > best_micro_f1:
            best_micro_f1 = test_micro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_micro_f1
            }, save_path)
            logging.info(f"Saved new best model with micro F1 score: {test_micro_f1:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_micro_f1'].append(test_micro_f1)
        history['test_macro_f1'].append(test_macro_f1)
        history['test_weighted_f1'].append(test_weighted_f1)
        history['main_loss'].append(epoch_main_loss)
        history['aux_loss'].append(epoch_aux_loss)
        history['contrastive_loss'].append(epoch_contrastive_loss)
        history['lr'].append(current_lr)
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {time_elapsed:.2f}s")
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Main Loss: {epoch_main_loss:.4f}, Aux Loss: {epoch_aux_loss:.4f}, Contrastive Loss: {epoch_contrastive_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Test Micro F1: {test_micro_f1:.4f}, Macro F1: {test_macro_f1:.4f}, Weighted F1: {test_weighted_f1:.4f}")
        logging.info(f"Epoch {epoch+1}/{num_epochs} - LR: {current_lr:.6f}")
        logging.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"Main Loss: {epoch_main_loss:.4f}, Aux Loss: {epoch_aux_loss:.4f}, Contrastive Loss: {epoch_contrastive_loss:.4f}")
        logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        logging.info(f"Test Micro F1: {test_micro_f1:.4f}, Macro F1: {test_macro_f1:.4f}, Weighted F1: {test_weighted_f1:.4f}")
        if epoch == 0 or epoch == 4 or epoch == 9 or (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            try:
                print(f"Visualizing features at epoch {epoch+1}...")
                # Uncomment if visualization is desired:
                # visualize_features(model, test_loader, num_samples=min(500, len(test_dataset)), 
                #                   epoch=epoch+1, name="test")
            except Exception as e:
                logging.error(f"Error in feature visualization: {e}")
                print(f"Error in feature visualization: {e}")
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            plot_training_curves(history, epoch+1)
    
    return history, best_micro_f1

# Function to plot and save training curves
def plot_training_curves(history, epochs):
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 2, 3)
    plt.plot(history['test_micro_f1'], label='Micro F1')
    plt.plot(history['test_macro_f1'], label='Macro F1')
    plt.plot(history['test_weighted_f1'], label='Weighted F1')
    plt.title('F1 Score Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 2, 4)
    plt.plot(history['main_loss'], label='Main Loss')
    plt.plot(history['aux_loss'], label='Auxiliary Loss')
    plt.plot(history['contrastive_loss'], label='Contrastive Loss')
    plt.title('Component Losses', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.subplot(3, 2, 5)
    plt.plot(history['lr'])
    plt.title('Learning Rate', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
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
    plt.savefig(f'./visualizations/training_curves_epoch_{epochs}.png', dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to './visualizations/training_curves_epoch_{epochs}.png'")

# Function to evaluate model and visualize results
def evaluate_model(model, test_loader, criterion, visualization_path='./visualizations'):
    model.eval()
    test_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    os.makedirs(visualization_path, exist_ok=True)
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating model"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, modal_outputs, modal_projections, global_projection, attention_weights = model(inputs, return_features=True)
            loss, _, _, _ = criterion(
                outputs, modal_outputs, modal_projections, global_projection, 
                targets, 0, 1
            )
            test_loss += loss.item() * inputs.size(0)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.vstack(all_probs) if all_probs else np.array([])
    test_acc = accuracy_score(all_targets, all_preds)
    test_micro_f1 = f1_score(all_targets, all_preds, average='micro')
    test_macro_f1 = f1_score(all_targets, all_preds, average='macro')
    test_weighted_f1 = f1_score(all_targets, all_preds, average='weighted')
    print(f"\nTest Results:")
    print(f"Loss: {test_loss/len(test_loader.dataset):.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Micro F1: {test_micro_f1:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print(f"Weighted F1: {test_weighted_f1:.4f}")
    logging.info(f"\nTest Results:")
    logging.info(f"Loss: {test_loss/len(test_loader.dataset):.4f}")
    logging.info(f"Accuracy: {test_acc:.4f}")
    logging.info(f"Micro F1: {test_micro_f1:.4f}")
    logging.info(f"Macro F1: {test_macro_f1:.4f}")
    logging.info(f"Weighted F1: {test_weighted_f1:.4f}")
    target_names = [f"Class {i}" for i in range(num_classes)]
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print("\nClassification Report:")
    print(report_df)
    logging.info(f"\nClassification Report:\n{report_df}")
    report_df.to_csv(f"{visualization_path}/classification_report.csv")
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_targets, all_preds)
    if num_classes > 20:
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
    plt.savefig(f"{visualization_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    if num_classes > 2:
        plt.figure(figsize=(12, 10))
        if num_classes > 10:
            display_classes = 10
            class_indices = np.random.choice(num_classes, display_classes, replace=False)
        else:
            display_classes = num_classes
            class_indices = range(num_classes)
        colors = plt.cm.rainbow(np.linspace(0, 1, display_classes))
        for i, class_idx in enumerate(class_indices):
            binary_targets = (all_targets == class_idx).astype(int)
            class_probs = all_probs[:, class_idx]
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
        plt.savefig(f"{visualization_path}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    try:
        tsne_embedding, tsne_labels = visualize_features(model, test_loader, num_samples=500, epoch="final", name="test")
    except Exception as e:
        logging.error(f"Error in TSNE visualization: {e}")
        print(f"Error in TSNE visualization: {e}")
    with torch.no_grad():
        inputs, _ = next(iter(test_loader))
        inputs = inputs.to(device)
        _, _, attention_weights = model(inputs)
        avg_attention = attention_weights.mean(0).cpu().numpy()
        plt.figure(figsize=(10, 6))
        modality_names = ['ESMC', 'ESMCStructure', 'OneHot', 'BLOSUM', 'GCN']
        avg_attention = avg_attention.squeeze() 
        plt.bar(modality_names, avg_attention, color='skyblue')
        plt.title('Average Modality Importance Weights', fontsize=16)
        plt.ylabel('Importance Weight', fontsize=14)
        plt.xticks(fontsize=12)
        plt.ylim(0, max(avg_attention) * 1.2)
        for i, v in enumerate(avg_attention):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{visualization_path}/modality_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Modality importance: {dict(zip(modality_names, avg_attention))}")
    return {
        'accuracy': test_acc,
        'micro_f1': test_micro_f1,
        'macro_f1': test_macro_f1,
        'weighted_f1': test_weighted_f1,
        'report': report_df
    }

def roc_curve(y_true, y_score):
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

def auc(x, y):
    return np.trapz(y, x)

if __name__ == "__main__":
    print("Starting enhanced multi-modal model training...")
    history, best_f1 = train_enhanced_model(
        model, train_val_loader, test_loader, criterion, optimizer, scheduler, 
        num_epochs=60, patience=10, save_path='./models/enhanced_transformer_best_model.pth'
    )
    checkpoint = torch.load('./models/enhanced_transformer_best_model.pth', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with F1 score {checkpoint['best_f1']:.4f}")
    results = evaluate_model(model, test_loader, criterion)
    with open('./models/model_config.txt', 'w') as f:
        f.write("Enhanced Multi-Modal Transformer\n")
        f.write("="*50 + "\n\n")
        f.write("Model Configuration:\n")
        f.write(f"Embedding Dimension: {embed_dim}\n")
        f.write(f"Number of Heads: {num_heads}\n")
        f.write(f"Number of Layers: {num_layers}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")
        f.write("Best Performance:\n")
        f.write(f"Best Epoch: {checkpoint['epoch']+1}\n")
        f.write(f"Best F1 Score: {checkpoint['best_f1']:.4f}\n\n")
        f.write("Final Evaluation Metrics:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Micro F1: {results['micro_f1']:.4f}\n")
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {results['weighted_f1']:.4f}\n")
    print("Training and evaluation completed successfully.")
    # Uncomment the following lines to perform cross-validation if desired
    # if False:
    #     print("\nPerforming cross-validation...")
    #     cv_results = perform_cross_validation(
    #         train_val_modality_tensors, y_train_val, 
    #         EnhancedMultiModalTransformer, num_folds=5, num_epochs=30, patience=10
    #     )
