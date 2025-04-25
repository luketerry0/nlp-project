from transformers import AutoModel, AutoTokenizer
from dataset import RiddleDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np

class RiddleSolverTransform():
    def __init__(self):
        pass

    def __call__(self, sample):
        # Apply transformation logic
        formatted_questions = []
        formatted_question = '[CLS] ' + sample['question']
        for i in range(4):
            formatted_questions.append(formatted_question +  ' [SEP] ' + sample['choice_list'][i] + ' [SEP]')

        return (formatted_questions, sample['label'])

class RiddleSolver(nn.Module):
    def __init__(self):
        super().__init__()
        # set up the model and the tokenizer
        MODEL_NAME = 'microsoft/deberta-v3-base'
        self.bert_model = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.dataset = RiddleDataset(transform=RiddleSolverTransform())
        indices = np.random.choice(len(self.dataset), 10, replace=False)
        self.subset = Subset(self.dataset, indices)
        self.dense_layer = torch.nn.Linear(768*4, 4)
        self.softmax = nn.Softmax()
        self.dataloader = DataLoader(self.subset, shuffle=True, batch_size=1)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)        # print(bert_model)
        # print(self.bert_model(**self.tokenizer(self.dataset[0]['formatted_question'], return_tensors="pt")))

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.softmax(x)
        return x
    
    def train(self):
        for epoch in range(2):  # loop over the dataset multiple times
            for data, labels in tqdm(self.dataloader):
                # pass each of the four answer options through the BERT model
                bert_questions = [
                    self.bert_model(**self.tokenizer(answer, return_tensors="pt"))['last_hidden_state'][0][0].detach()
                    for answer in data
                    ]
                # print("-=-")
                # print(bert_questions[0].shape)
                # print(bert_questions[1].shape)
                # print(bert_questions[2].shape)
                # print(bert_questions[3].shape)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(torch.cat(bert_questions))
                correct_output = torch.zeros(4)
                correct_output[labels] = 1
                loss = self.criterion(outputs, correct_output)
                loss.backward()
                self.optimizer.step()


            print('Finished Training')


net = RiddleSolver()
net.train()


