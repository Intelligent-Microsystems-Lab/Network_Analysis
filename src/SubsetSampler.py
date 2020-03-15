import torch
from torch.utils.data.sampler import Sampler

class SubsetSampler(Sampler):
    """
    This Sampler only samples from elements of a specified class.
    Inputs: data_source-The dataset to sample from
            label-The value of the label to filter by
            n-optional argument to prune the samples gathered
    """
    def __init__(self,data_source,label,n=-1):
        self.data_source = data_source
        self.label = label
        self.n = n

    def __iter__(self):
        return iter([i for i in range(len(self.data_source)) if self.data_source.__getitem__(i)[1] == self.label][:self.n])

    def __len__(self):
        return len(self.data_source)
