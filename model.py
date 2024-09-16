from torch import nn
import torch.nn.functional as F
import torch
import math

def _generate_positional_encoding(max_sequence_length, d_model):
        positional_encoding = torch.zeros(max_sequence_length, d_model)
        position = torch.arange(0, max_sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding.unsqueeze(0)

class TaskHead(nn.Module):
    """
    Designed to address sequential inputs"""
    def __init__(self, hidden_dim, n_classes, dropout=0.2):
        super(TaskHead, self).__init__()
        self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, n_classes),
                nn.Softmax(dim=-1)
            )

    def forward(self, x):
        out = self.mlp(x)
        out = out.max(dim=1)[0]
        return out

class SingleTaskModel(nn.Module):
    def __init__(self, num_features, n_classes=2, num_heads=8, num_layers=2, dropout=0.2):
        super(SingleTaskModel, self).__init__()
        self.hidden_dim = num_features
        self.num_segments = 10
        transformer_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                       nhead=num_heads,
                                                       dim_feedforward=4096,
                                                       batch_first=True, dropout=dropout)
        
        self.encoder = nn.TransformerEncoder(transformer_layer, num_layers)
        self.positional_encoding = _generate_positional_encoding(self.num_segments, num_features).cuda()
        self.task_head = TaskHead(self.hidden_dim, n_classes, dropout=dropout)

    def forward(self, x):
        x = x + self.positional_encoding
        enc = self.encoder(x)
        out = self.task_head(enc)
        return out


class MultiTaskModelTaskSpecificAttn(nn.Module):
    def __init__(self, num_features, n_classes=(2, 4), num_heads=8, num_layers=2, dropout=0.2):
        super(MultiTaskModelTaskSpecificAttn, self).__init__()
        self.hidden_dim = num_features
        self.num_segments = 10
        transformer_layerA = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                       nhead=8,
                                                       dim_feedforward=4096,
                                                       batch_first=True, dropout=dropout)
        
        transformer_layerB = nn.TransformerEncoderLayer(d_model=self.hidden_dim, 
                                                       nhead=8,
                                                       dim_feedforward=4096,
                                                       batch_first=True, dropout=dropout)
      
        
        self.transformerA = nn.TransformerEncoder(transformer_layerA, 2)
        self.transformerB = nn.TransformerEncoder(transformer_layerB, 2)

        self.positional_encoding = _generate_positional_encoding(self.num_segments, num_features).cuda()
    
        # Task-Specific Attention
        self.taskAB = nn.MultiheadAttention(embed_dim=num_features, num_heads=num_heads, batch_first=True)
        self.taskBA = nn.MultiheadAttention(embed_dim=num_features, num_heads=num_heads, batch_first=True)
      
        self.wander_head = TaskHead(self.hidden_dim, n_classes[0])
        self.engage_head = TaskHead(self.hidden_dim, n_classes[1])

    
    def forward(self, x):
        x = x + self.positional_encoding
        # Task-Specific Attention
        outA_weighted, _ = self.taskAB(x, x, x)
        outB_weighted, _ = self.taskBA(x, x, x)

        out = self.transformerA(x)

        out_wander = self.wander_head(out) + self.wander_head(self.transformerB(outA_weighted))
        out_engage = self.engage_head(out) + self.engage_head(self.transformerB(outB_weighted))
        
        return out_wander, out_engage