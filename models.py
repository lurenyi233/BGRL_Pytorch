from torch_geometric.nn import GCNConv

import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

import copy

"""
The following code is borrowed from BYOL, SelfGNN
and slightly modified for BGRL
"""

# 计算指数移动平均（Exponential Moving Average，EMA）
class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

# 衡量两个向量之间的余弦相似度
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# 用指数移动平均 (EMA) 更新两个模型的参数
def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# 是否计算梯度
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        
        self.conv1 = GCNConv(layer_config[0], layer_config[1])
        self.bn1 = nn.BatchNorm1d(layer_config[1], momentum = 0.01)
        self.prelu1 = nn.PReLU()
        self.conv2 = GCNConv(layer_config[1],layer_config[2])
        self.bn2 = nn.BatchNorm1d(layer_config[2], momentum = 0.01)
        self.prelu2 = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(self.bn2(x))

        return x

# 这是一个用于初始化神经网络模型权重的函数。
# 函数通过遍历模型的每个模块（m），如果模块是线性层（nn.Linear），
# 则使用 Xavier（也称为Glorot）初始化方法初始化权重，并将偏置项初始化为常数值。
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BGRL(nn.Module):

    def __init__(self, layer_config, pred_hid, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()
        self.student_encoder = Encoder(layer_config=layer_config, dropout=dropout, **kwargs)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)  # 深度拷贝，以确保两者是相互独立的对象。
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)
        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)
    
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x1, x2, edge_index_v1, edge_index_v2, edge_weight_v1=None, edge_weight_v2=None):
        v1_student = self.student_encoder(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
        v2_student = self.student_encoder(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)

        v1_pred = self.student_predictor(v1_student)
        v2_pred = self.student_predictor(v2_student)
        
        with torch.no_grad():
            v1_teacher = self.teacher_encoder(x=x1, edge_index=edge_index_v1, edge_weight=edge_weight_v1)
            v2_teacher = self.teacher_encoder(x=x2, edge_index=edge_index_v2, edge_weight=edge_weight_v2)
            
        loss1 = loss_fn(v1_pred, v2_teacher.detach())
        loss2 = loss_fn(v2_pred, v1_teacher.detach())

        loss = loss1 + loss2
        return v1_student, v2_student, loss.mean()


class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):

        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)

        return logits, loss