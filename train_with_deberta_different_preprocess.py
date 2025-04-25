from transformers import AutoModel, AutoTokenizer
from dataset import RiddleDataset
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Subset
import wandb
import math
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def main(args):
    class EditedTransform():
        def __init__(self):
            pass

        def __call__(self, sample):
            # Apply transformation logic
            formatted_question = '[CLS] ' + sample['question']
            for i in range(4):
                formatted_question += ' [SEP] ' + sample['choice_list'][i]

            return (formatted_question, sample['label'])

    class RiddleSolver(nn.Module):
        def __init__(self):
            super().__init__()
            # set up the model and the tokenizer
            MODEL_NAME = 'microsoft/deberta-v3-base'
            self.bert_model = AutoModel.from_pretrained(MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.full_dataset = RiddleDataset(transform=EditedTransform())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_proportion = 0.1
            self.train_set, self.test_set = torch.utils.data.random_split(
                self.full_dataset, 
                [math.ceil(len(self.full_dataset)*(1-test_proportion)), math.floor(len(self.full_dataset)*(test_proportion))]
                )

            # code to use a subset of the data (for local testing)
            indices = np.random.choice(len(self.train_set), 10, replace=False)
            self.train_subset = Subset(self.train_set, indices)


            self.train_loader = DataLoader(self.train_subset, shuffle=True, batch_size=args.batch_size)
            self.test_loader = DataLoader(self.test_set, shuffle=True, batch_size=args.batch_size)
            self.dense_layer = torch.nn.Linear(768, 4).to(self.device)
            self.softmax = nn.Softmax().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.SGD(self.parameters(), lr=args.learning_rate, momentum=0.9)        # print(bert_model)
            # print(self.bert_model(**self.tokenizer(self.dataset[0]['formatted_question'], return_tensors="pt")))
            self.epochs = args.epochs

        def forward(self, x):
            x = self.dense_layer(x)
            x = self.softmax(x)
            return x
        
        def evaluate(self, epoch):
            y_pred = []
            y_true = []

            with torch.no_grad():
                for data, labels in tqdm(self.test_loader):

                    # pass each of the four answer options through the BERT model
                    bert_question = self.bert_model(
                        **self.tokenizer(data[0], return_tensors="pt")
                        )['last_hidden_state'][0][0].detach()

                    # forward pass
                    inp = bert_question.to(self.device)
                    outputs = self(inp)
                    correct_output = torch.zeros(4).cpu()
                    correct_output[labels] = 1

                    y_pred.append(int(np.argmax(outputs.detach().cpu())))
                    y_true.append(int(np.argmax(correct_output)))

                    del inp
                    del outputs
                    del correct_output
        
            report = classification_report(y_true, y_pred, output_dict=True)
            matrix = confusion_matrix(y_true, y_pred)
            wandb.log(data=report, step=epoch)
            wandb.log(data={'confusion_matrix': matrix}, step=epoch)
    
        
        def train(self):
            for epoch in range(self.epochs): 
                for data, labels in tqdm(self.train_loader):
                    
                    # pass each of the four answer options through the BERT model
                    bert_question = self.bert_model(
                        **self.tokenizer(data[0], return_tensors="pt")
                        )['last_hidden_state'][0][0].detach()

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    inp = bert_question.to(self.device)                    
                    outputs = self(inp)
                    correct_output = torch.zeros(4).to(self.device)
                    correct_output[labels] = 1
                    loss = self.criterion(outputs, correct_output)
                    loss.backward()
                    self.optimizer.step()
                    del inp
                    del outputs

                self.evaluate(epoch)

                print('Finished Training')

    net = RiddleSolver()
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='epochs to train for')

    args = parser.parse_args()

    # login to wandb
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="luketerry0-university-of-oklahoma",
    # Set the wandb project where this run will be logged.
    project="modernbert-project-gpu-test",
    # Track hyperparameters and run metadata.
    config=args,
    )

    main(args)