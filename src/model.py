import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv,   
    GlobalAttention,
    GCNConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool
)

from typing import List
from src.loss import ce_loss

class UDD(nn.Module):
    def __init__(self, 
                 classifiers, 
                 num_classes:int=2,
                 lambda_epochs:int=1):
        
        super().__init__()
        self.views = len(classifiers)
        self.lambda_epochs = lambda_epochs
        self.classes = num_classes
        self.classifiers = classifiers

    def DS_Combin(self, alpha):

        def DS_Combin_two(alpha1, alpha2):

            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]

            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))


            S_a = self.classes / u_a
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a, u_a

        if len(alpha) == 1:
            u_a = self.classes/torch.sum(alpha[0], dim=1, keepdim=True)
            return alpha[0], u_a

        for v in range(len(alpha)-1):
            if v == 0:
                alpha_a, u_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a, u_a
    
    def infer(self, X):

        evidence = dict()
        for v_num in range(self.views):
            # print(X[v_num].dtype)
            evidence[v_num] = self.classifiers[v_num](X[v_num])
        return evidence

    def forward(self, X, y, global_step):
        X = list(X)
        evidence = self.infer(X)
        loss = 0
        alpha = dict()
        for v_num in range(len(X)):
            alpha[v_num] = evidence[v_num] + 1
            loss += ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs)

        alpha_a, u_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += 0.3 * ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence, evidence_a, loss, u_a

class TabularBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.35):
        super().__init__()
        hidden_dim = dim * expansion
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class TabularClassifer(nn.Module):
    def __init__(self, 
                 input_dim:int=3, 
                 hidden_dim:int=64, 
                 num_classes:int=2,
                 dropout:float=0.35,
                 softplus:bool=False):
        
        super().__init__()
        bottleneck_dim = max(hidden_dim // 2, 16)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            TabularBlock(hidden_dim, expansion=2, dropout=dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        
        self.softplus = softplus
        
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        
        if self.softplus:
            return F.softplus(x)
        
        return F.log_softmax(x, dim=1)

class GNNClassifier(nn.Module):
    def __init__(self, 
                 node_feature_dim:int=5, 
                 hidden_dim:int=128, 
                 num_classes:int=2, 
                 dropout:float=0.6,
                 softplus:bool=False):
        super(GNNClassifier, self).__init__()
        
        self.conv1 = SAGEConv(node_feature_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.dropout = dropout
        self.softplus = softplus

        self.pool = global_mean_pool

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.norm1(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_res = F.leaky_relu(self.conv2(x, edge_index))
        x_res = self.norm2(x_res, batch)
        x = x + x_res  # residual connection

        x = self.pool(x, batch)
        x = self.classifier(x)
        
        if self.softplus:
            return F.softplus(x)
        
        return F.log_softmax(x, dim=1)
