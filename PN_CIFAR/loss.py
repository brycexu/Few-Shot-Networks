from torch.nn import functional as F
import torch

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def loss_fn(input, target, n_support):
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
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    return loss_val, acc_val