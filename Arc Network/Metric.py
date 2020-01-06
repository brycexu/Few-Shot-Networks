# Author : Bryce Xu
# Time : 2019/12/12
# Function: Metric

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class ArcMarginProduct(nn.Module):

    def __init__(self, in_features, out_features, s=30, m=0.5, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)
        output *= self.s
        return output

def cosine_dist(x, y):
    mul = torch.mm(x, y.t())
    res = mul / (x.norm() * y.norm() + 1e-5)
    return res

def eval(input, target, n_support):
    input_cpu = input.to('cpu')
    target_cpu = target.to('cpu')
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    # target 从小到大排列
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    # 在 Support Set 中按照 target 聚类, 这里每个聚类有5个样本的 index
    support_idxs = list(map(supp_idxs, classes))
    # 得到每个聚类的 Prototype
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # 在 Support Set 中找出 Query 样本的 index
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input_cpu[query_idxs]
    output = cosine_dist(query_samples, prototypes)
    softmax = F.softmax(output, dim=1).view(n_classes, n_query, -1)
    _, predicted_index = softmax.max(2)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1)
    correct = predicted_index.eq(target_inds).sum().item()
    return correct