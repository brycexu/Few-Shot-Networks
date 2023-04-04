import numpy as np
import torch

class BatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, n_iter, n_way, n_shot, n_query):
        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.classes = list(np.sort(np.unique(np.array(labels))))
        self.classes_to_idxes = []
        for i in self.classes:
            self.classes_to_idxes.append(torch.from_numpy(np.argwhere(labels == i).reshape(-1)))

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            selected_classes = torch.randperm(len(self.classes))[:self.n_way]
            support_set = []
            query_set = []
            for c in selected_classes:
                idxes = self.classes_to_idxes[c.item()]
                pos = torch.randperm(idxes.size()[0])
                support_set.append(idxes[pos[:self.n_shot]])
                query_set.append(idxes[pos[self.n_shot:self.n_shot+self.n_query]])
            batch = torch.cat(support_set+query_set)
            yield batch
