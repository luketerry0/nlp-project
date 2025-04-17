"""
This file contains a custom torch dataset which we can use to load the data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RiddleDataset(Dataset):
    """Dataset for the problem set"""

    def __init__(self, sp=True, train=True, root_dir="/home/cs529321/nlp-project/data/", transform=None):
        dataset = "SP" if sp else "WP"
        if train:
            self.data = np.load(root_dir+dataset+"-train.npy", allow_pickle=True)
        else:
            self.data = np.load(root_dir+dataset+"_eval_data_for_practice.npy", allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.data[idx])
        else:
            sample = self.data[idx]
        return sample

class CustomTransform:
    def __init__(self):
        pass

    def __call__(self, sample):
        # Apply transformation logic
        formatted_question = '[CLS] ' + sample['question']
        for i in range(4):
            formatted_question += ' [SEP] ' + sample['choice_list'][i]

        return (formatted_question, sample['label'])

if __name__ == "__main__":
    test = RiddleDataset(transform=CustomTransform())
    print(len(test))
    print(test[0])