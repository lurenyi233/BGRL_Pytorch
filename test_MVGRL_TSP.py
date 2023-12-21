from tsp50 import TSPDataset50
from torch_geometric.loader import DataLoader

import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch_geometric.transforms as T

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, LREvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, data):
        x, edge_index, edge_attr, edge_label_index = data.pos, data.edge_index, data.edge_attr, data.edge_label_index
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        z1 = self.encoder1(x1, edge_index1, edge_weight1)
        z2 = self.encoder2(x2, edge_index2, edge_weight2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1, edge_weight1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2, edge_weight2))
        return z1, z2, g1, g2, z1n, z2n


def train(encoder_model, contrast_model, train_loader, optimizer, device):
    encoder_model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        z1, z2, g1, g2, z1n, z2n = encoder_model(data)
        loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
        loss.backward()
        optimizer.step()
    total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


def test(encoder_model, loader, device):
    encoder_model.eval()

    true_labels = []
    predicted_labels = []
    total_loss = 0

    for data in loader:
        data = data.to(device)

        z1, z2, _, _, _, _ = encoder_model(data)
        z = z1 + z2
        split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
    result = LREvaluator()(z, data.y, split)
    return result


cdist = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for i in range(0, 1):
    device = torch.device('cpu')

    dataset = TSPDataset50(root="TSPDataset50", name='TSP', split='train')

    train_dataset = dataset[0 + i * 833:666 + i * 833]
    val_dataset = dataset[666 + i * 833:749 + i * 833]
    test_dataset = dataset[749 + i * 833:833 + i * 833]
    print(cdist[i], len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    aug1 = A.Identity()
    aug2 = A.PPRDiffusion(alpha=0.2)
    gconv1 = GConv(input_dim=2, hidden_dim=8, num_layers=2).to(device)
    gconv2 = GConv(input_dim=2, hidden_dim=8, num_layers=2).to(device)
    encoder_model = Encoder(encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2), hidden_dim=8).to(device)

    # 计算 Jensen-Shannon Distance 距离的损失函数
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L').to(device)

    optimizer = Adam(encoder_model.parameters(), lr=0.001)