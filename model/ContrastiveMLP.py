import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossModalAttention_1123(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # x shape = [B, 3, 256]
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(attn_out + x)
        x = self.norm(self.ff(x) + x)
        return x
class MoleculeEncoder_1123(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.cross_attn = CrossModalAttention_1123(dim)
        self.pool = nn.Linear(dim, 1)  # attention pooling

    def forward(self, gnn_feats):
        B = gnn_feats.shape[0]
        x = gnn_feats.view(B, 3, -1)
        x = self.cross_attn(x)

        w = torch.softmax(self.pool(x), dim=1)
        x = (x * w).sum(dim=1)
        return x

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_dim, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_dim, eps=eps)
        self.gate = nn.Parameter(torch.tensor(-5.0))
    def forward(self, x):
        ln_out = self.ln(x)
        alpha=torch.sigmoid(self.gate)
        out = (1-alpha)*x + alpha*ln_out
        return out

class DoubleMLP_V1212_gnnrep(nn.Module):
    def __init__(self, input_dim=1792, embed_dim=512, dropout=0.3):
        super().__init__()
        self.mol_encoder = MoleculeEncoder_1123(dim=256)
        # self.fusion_encoder = FusionEncoder_1123(dim=embed_dim)
        self.encoder1 = nn.Sequential(
            nn.Linear(1280+1024+256, embed_dim * 4),
            nn.BatchNorm1d(embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # 对 input2（相似对拼接特征）的编码器
        self.encoder2 = nn.Sequential(
            nn.Linear(1280+1024+256, embed_dim * 4),
            nn.BatchNorm1d(embed_dim * 4),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shared_layer=nn.Linear(embed_dim*4,embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2 , 120),
            AdaptiveLayerNorm(120),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(120, 2)
        )
        self.gate_weight = nn.Linear(embed_dim, 10)
        nn.init.constant_(self.gate_weight.bias, -1.0)
    def forward(self, input1_ecfp,input1_gnn, input2_ecfp,input2_gnn):
        batch_size = input1_ecfp.size(0)

        gnn_raw = input1_gnn[..., 1280:]
        gnn_feat = self.mol_encoder(gnn_raw)
        input1 = torch.cat([input1_ecfp, gnn_feat], dim=-1)

        z1 = self.shared_layer(self.encoder1(input1))
        z1_norm = F.normalize(z1, dim=-1)

        input2_ecfp = input2_ecfp.view(batch_size, 10, 2304)
        input2_gnn = input2_gnn.view(batch_size, 10, 1280+256*3)
        gnn2_raw = input2_gnn[..., 1280:]

        gnn2_flat = gnn2_raw.reshape(batch_size * 10, -1)
        gnn2_feat = self.mol_encoder(gnn2_flat)
        gnn2_feat = gnn2_feat.view(batch_size, 10, -1)
        input2 = torch.cat([input2_ecfp, gnn2_feat], dim=-1)

        z2_i = [self.shared_layer(self.encoder2(input2_i)) for input2_i in torch.unbind(input2, dim=1)]
        z2_all = torch.stack(z2_i, dim=1)
        z2_all_norm = F.normalize(z2_all, dim=-1)

        gate_weights = torch.sigmoid(self.gate_weight(z1))
        weighted_z2 = (z2_all * gate_weights.unsqueeze(-1)).sum(dim=1)
        x = torch.cat((z1, weighted_z2), dim=1)
        out = self.classifier(x)
        return z1_norm, z2_all_norm, out
