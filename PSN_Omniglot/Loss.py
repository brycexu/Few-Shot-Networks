# Author : Bryce Xu
# Time : 2019/11/18
# Function: 损失函数

import torch
from torch.nn import functional as F

def euclidean_dist(x, y, z):
    n = x.size(0) # 300
    m = y.size(0) # 60
    d = x.size(1) # 64
    x = x.unsqueeze(1).expand(n, m, d) # (300,1,64) -> (300,60,64)
    y = y.unsqueeze(0).expand(n, m, d) # (1,60,64) -> (300,60,64)
    z = z.unsqueeze(0).expand(n, m, d, d) # (60,64,64) -> (300,60,64,64)
    e = x - y # (300,60,64)
    e2 = e.unsqueeze(3) # (300,60,64,1)
    ze = torch.matmul(z, e2) # (300,60,64,1)
    ze = ze.squeeze(3)
    return torch.pow(e-ze, 2).sum(2)

def loss_fn(input, target, n_support):
    input_cpu = input.to('cpu')
    target_cpu = target.to('cpu')
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    means, projectives = get_prototypes(input_cpu, support_idxs)
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, means, projectives)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val

def get_prototypes(input_cpu, support_idxs, num_subspaces=3):
    support_embeddings = []
    for support_idx in support_idxs:
        support_embeddings.append(torch.stack([input_cpu[idx_list] for idx_list in support_idx]))
    support_embeddings = torch.stack(support_embeddings, dim=0)
    # [60, 5, 64]
    means = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # [60, 64]
    support_embeddings -= means.unsqueeze(1)
    projectives = []
    for class_idx in range(support_embeddings.size(0)):
        u, s, v = torch.svd(support_embeddings[class_idx], some=False)
        projectives.append(torch.mm(v[:, :num_subspaces], v[:, :num_subspaces].T))
    projectives = torch.stack(projectives)
    # [60, 64, 64]
    return means, projectives

def euclidean_dist2(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def loss_fn2(input, target, n_support):
    input_cpu = input.to('cpu')
    target_cpu = target.to('cpu')
    def supp_idxs(c):
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support
    support_idxs = list(map(supp_idxs, classes))
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist2(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val