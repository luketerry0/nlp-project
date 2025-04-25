"""
This file contains a custom torch dataset which we can use to load the data
"""

import torch
import torch.utils.data
import numpy as np
from datasets import Dataset

class RiddleDataset(torch.utils.data.Dataset):
    """Dataset for the problem set"""

    def __init__(self, sp=True, root_dir="./data/", transform=None):
        dataset = "SP" if sp else "WP"
        self.d = np.load(root_dir+dataset+"-train.npy", allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return self.d.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.d[idx])
        else:
            sample = self.d[idx]
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
    
# get the pytorch dataset as a generator in order to make a huggingface dataset with it
def get_huggingface_dataset(sp=True, root_dir="./data/"):
    dataset = "SP" if sp else "WP"
    data = np.load(root_dir+dataset+"-train.npy", allow_pickle=True)
    def data_gen(data):
        # you need to manually set the type of all features or errors happen
        for i in range(0, len(data)):
            # Apply transformation logic
            formatted_question = '[CLS] ' + data[i]['question']
            for j in range(4):
                formatted_question += ' [SEP] ' + data[i]['choice_list'][j]

            yield {
                'formatted_question': str(formatted_question),
                'id': str(data[i]['id']),
                'question': str(data[i]['question']),
                'label': str(data[i]['label']),
                'choice_list': [str(choice) for choice in data[i]['choice_list']],
                'label': int(data[i]['label'])
                }

    ds = Dataset.from_generator(data_gen, gen_kwargs={"data": data})
    return ds
    

if __name__ == "__main__":
    ds = RiddleDataset(transform=CustomTransform())
    print(len(ds))
    print(ds[0])
    dl = torch.utils.data.DataLoader(ds)
    for i in dl:
        print(i)

    # huggingface_dataset = get_huggingface_dataset(train=True, sp=False)
    # print(huggingface_dataset[0])
