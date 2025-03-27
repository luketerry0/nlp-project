"""
This file contains a custom torch dataset which we can use to load the data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RiddleDataset(Dataset):
    """Dataset for the problem set"""

    def __init__(self, sp=True, train=True, root_dir="/home/cs529321/nlp-project/data/"):
        dataset = "SP" if sp else "WP"
        if train:
            self.data = np.load(root_dir+dataset+"-train.npy", allow_pickle=True)
        else:
            self.data = np.load(root_dir+dataset+"_eval_data_for_practice.npy", allow_pickle=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    test = RiddleDataset()
    print(len(test))
    print(test[0])