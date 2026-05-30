import numpy as np
import torch
from torch.utils.data import Sampler
import math
import random


class BalancedBatchSampler_0925(Sampler):
    def __init__(self, labels, batch_size, pos_fraction=0.5, drop_last=False):
        self.labels = np.asarray(labels)
        assert set(np.unique(self.labels)).issubset({0,1}), "labels must be 0/1"
        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        self.n_pos = int(round(self.batch_size * self.pos_fraction))
        self.n_neg = self.batch_size - self.n_pos
        self.drop_last = drop_last

        self.pos_indices = np.where(self.labels == 1)[0].tolist()
        self.neg_indices = np.where(self.labels == 0)[0].tolist()

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("Need both positive and negative samples in labels.")

        if drop_last:
            self.num_batches = len(self.labels) // self.batch_size
        else:
            self.num_batches = math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        pos_pool = self.pos_indices.copy()
        neg_pool = self.neg_indices.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)

        pos_repeat = math.ceil(self.num_batches * self.n_pos / len(pos_pool))
        neg_repeat = math.ceil(self.num_batches * self.n_neg / len(neg_pool))

        pos_pool = (pos_pool * pos_repeat)[:self.num_batches * self.n_pos]
        neg_pool = (neg_pool * neg_repeat)[:self.num_batches * self.n_neg]

        for i in range(self.num_batches):
            pos_batch = pos_pool[i*self.n_pos:(i+1)*self.n_pos]
            neg_batch = neg_pool[i*self.n_neg:(i+1)*self.n_neg]
            batch_idx = pos_batch + neg_batch
            random.shuffle(batch_idx)
            yield batch_idx

    def __len__(self):
        return self.num_batches

class BalancedBatchSampler_1011CV(Sampler):

    def __init__(self, indexs,labels, batch_size, pos_fraction=0.5, drop_last=False):
        self.index=np.asarray(indexs)
        self.labels = np.asarray(labels)
        assert set(np.unique(self.labels)).issubset({0,1}), "labels must be 0/1"
        self.batch_size = int(batch_size)
        self.pos_fraction = float(pos_fraction)
        self.n_pos = int(round(self.batch_size * self.pos_fraction))
        self.n_neg = self.batch_size - self.n_pos
        self.drop_last = drop_last

        self.pos_indices = np.where(self.labels == 1)[0].tolist()
        self.neg_indices = np.where(self.labels == 0)[0].tolist()

        if len(self.pos_indices) == 0 or len(self.neg_indices) == 0:
            raise ValueError("Need both positive and negative samples in labels.")
        if drop_last:
            self.num_batches = len(self.labels) // self.batch_size
        else:
            self.num_batches = math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):

        pos_pool = self.pos_indices.copy()
        neg_pool = self.neg_indices.copy()
        random.shuffle(pos_pool)
        random.shuffle(neg_pool)

        pos_repeat = math.ceil(self.num_batches * self.n_pos / len(pos_pool))
        neg_repeat = math.ceil(self.num_batches * self.n_neg / len(neg_pool))

        pos_pool = (pos_pool * pos_repeat)[:self.num_batches * self.n_pos]
        neg_pool = (neg_pool * neg_repeat)[:self.num_batches * self.n_neg]

        for i in range(self.num_batches):
            pos_batch = pos_pool[i*self.n_pos:(i+1)*self.n_pos]
            neg_batch = neg_pool[i*self.n_neg:(i+1)*self.n_neg]
            batch_idx = pos_batch + neg_batch
            random.shuffle(batch_idx)
            yield batch_idx

    def __len__(self):
        return self.num_batches