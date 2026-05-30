import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, MessagePassing, GINEConv
from torchvision.models import resnet50, ResNet50_Weights, resnet34, ResNet34_Weights, resnet18, ResNet18_Weights
from torch_geometric.utils import add_self_loops

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x=F.normalize(x, dim=-1)
        return x

class EdgeMPNN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr, pos):
        row, col = edge_index
        dist = torch.norm(pos[row] - pos[col], p=2, dim=1).unsqueeze(-1)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, dist=dist)

    def message(self, x_j, edge_attr, dist):
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        msg = torch.cat([x_j, edge_attr, dist], dim=-1)
        return self.node_mlp(msg)

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))

class MolGraphEncoder_1111(nn.Module):
    def __init__(self, node_dim=132, edge_dim=6, hidden_dim=256, out_dim=512, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            EdgeMPNN(node_dim if i == 0 else hidden_dim, edge_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.res_proj = nn.Linear(node_dim, hidden_dim)
    def forward(self, data):
        x, edge_index, edge_attr, pos, batch = data.x, data.edge_index, data.edge_attr, data.pos, data.batch
        edge_index, edge_attr = add_self_loops(
            edge_index,
            edge_attr=edge_attr,
            fill_value=0,
            num_nodes=x.size(0)
        )
        for layer_idx, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x_res = x
            x = conv(x, edge_index, edge_attr, pos)
            if layer_idx == 0:
                x = norm(x + self.res_proj(x_res))
            else:
                x = norm(x + x_res)
        x = global_mean_pool(x, batch)
        h = F.normalize(self.readout(x), dim=-1)
        return h
class Multimodal_KM_1117(nn.Module):
    def __init__(self, graph_in_dim, graph_hidden_dim, proj_dim=512,fusion_out_dim=128):
        super().__init__()
        self.graph_encoder = MolGraphEncoder_1111(node_dim=132, edge_dim=6, hidden_dim=graph_hidden_dim, out_dim=512, num_layers=5)
        self.image_encoder = ImageEncoder()
        self.graph_proj = nn.Sequential(
            nn.Linear(512, proj_dim*2),
            nn.BatchNorm1d(proj_dim*2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(proj_dim*2, proj_dim)
        )

        self.encoder_enzyme = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
        )

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim +256, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, graph_data, img_2d, img_3d,esm):
        esm = esm.squeeze(1)
        batch_size, num_views, C, H, W = img_3d.shape
        img_flat = img_3d.view(batch_size * num_views, C, H, W)

        graph_rep = self.graph_encoder(graph_data)
        img2d_rep = self.image_encoder(img_2d)
        view_features = self.image_encoder(img_flat)
        view_features = view_features.view(batch_size, num_views, -1)
        img3d_rep=view_features.mean(dim=1)
        z_graph = self.graph_proj(graph_rep)
        z_2d = self.graph_proj(img2d_rep)
        z_3d = self.graph_proj(img3d_rep)
        fused_rep = torch.mean(torch.stack([z_graph, z_2d, z_3d]), dim=0)

        enzyme=self.encoder_enzyme(esm)
        x=torch.cat([enzyme,fused_rep],dim=1)
        out=self.classifier(x)
        return z_graph, z_2d, z_3d, fused_rep,out
    def get_GNN_rep(self, graph_data, img_2d, img_3d):
        batch_size, num_views, C, H, W = img_3d.shape
        img_flat = img_3d.view(batch_size * num_views, C, H, W)
        graph_rep = self.graph_encoder(graph_data)
        img2d_rep = self.image_encoder(img_2d)
        view_features = self.image_encoder(img_flat)
        view_features = view_features.view(batch_size, num_views, -1)
        img3d_rep = view_features.mean(dim=1)
        z_graph = self.graph_proj(graph_rep)
        z_2d = self.graph_proj(img2d_rep)
        z_3d = self.graph_proj(img3d_rep)
        fused_rep = torch.mean(torch.stack([z_graph, z_2d, z_3d]), dim=0)
        cat_rep=torch.cat([z_graph,z_2d,z_3d],dim=-1)
        return z_graph, z_2d, z_3d, fused_rep,cat_rep

if __name__ == '__main__':
    img_model=ImageEncoder(256)
    print(img_model)
