from transformers import BertModel, AutoTokenizer, AutoModel, get_scheduler

from dataset import RiddleDataset
import torch.nn as nn
from torch.optim import AdamW
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
            super(RiddleSolver, self).__init__()
            # set up the model and the tokenizer
            MODEL_NAME = 'answerdotai/ModernBERT-base'
            self.batch_size=args.batch_size
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.full_dataset = RiddleDataset(transform=EditedTransform())
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
            test_proportion = 0.1
            self.train_set, self.test_set = torch.utils.data.random_split(
                self.full_dataset, 
                [math.ceil(len(self.full_dataset)*(1-test_proportion)), math.floor(len(self.full_dataset)*(test_proportion))]
                )
            
            #subset for prototyping
            subset_size = 10
            subset_indices = np.random.choice(len(self.train_set), size=subset_size, replace=False)
            self.train_set = torch.utils.data.Subset(self.train_set, subset_indices)


            self.train_loader = DataLoader(self.train_set, shuffle=True, batch_size=args.batch_size)
            self.test_loader = DataLoader(self.test_set, shuffle=True, batch_size=1)
            self.dense_layer = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4)
            ).to(self.device)
            self.softmax = nn.Softmax().to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = AdamW([
                {"params": self.bert_model.parameters(), "lr": args.learning_rate},
                {"params": self.dense_layer.parameters(), "lr": 1e-2}
            ], weight_decay=0.01)            
            self.epochs = args.epochs
            total_steps = len(self.train_loader) * self.epochs
            self.lr_scheduler = get_scheduler(
                name="linear",
                optimizer=self.optimizer,
                num_warmup_steps=int(0.1 * total_steps),  # warm up for first 10% of steps
                num_training_steps=total_steps,
            )
            
            # print some stuff about the data
            print(f"Train set size: {len(self.train_set)}")
            print(f"Test set size: {len(self.test_set)}")
            print(f"Batch size: {self.batch_size}")


        def forward(self, ids, mask, sm=False):
            output = self.bert_model(
                ids, attention_mask=mask
            )
            x = self.dense_layer(output.last_hidden_state[:, 0, :]) # extract the first token...
            x = x.squeeze(-1)
            if sm: 
                x = self.softmax(x)
            return x
        
        def evaluate(self, epoch):
            y_pred = []
            y_true = []

            with torch.no_grad():
                self.bert_model.eval()
                for batch_question, batch_labels in tqdm(self.test_loader):

                    #tokenize flattened choices
                    encoding = self.tokenizer(
                        batch_question,
                        return_tensors='pt',
                        padding=True,
                        add_special_tokens=False
                    )

                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    outputs = self(input_ids, mask=attention_mask, sm=True)                     
                    outputs = outputs.view(len(batch_labels), 4)
                    labels = torch.tensor(batch_labels).to(self.device) 
                    y_pred.append(torch.argmax(outputs, dim=1).cpu())
                    y_true.append(labels.cpu())

            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            print(y_true)
            print(y_pred)
            report = classification_report(y_true, y_pred, output_dict=True)
            matrix = confusion_matrix(y_true, y_pred)
            print(report)
            wandb.log(data=report, step=epoch)
            wandb.log(data={'confusion_matrix': matrix}, step=epoch)
    

        def train(self):
            print("beginning training")
            for epoch in tqdm(range(self.epochs)): 
                self.bert_model.train()

                if epoch < 5:  # freeze DeBERTa 
                    for param in self.bert_model.parameters():
                        param.requires_grad = False
                else:
                    for param in self.bert_model.parameters():
                        param.requires_grad = True

                total_loss=0
                num_batches = 0
                
                for batch_question, batch_labels in tqdm(self.train_loader):

                    #tokenize flattened choices
                    encoding = self.tokenizer(
                        batch_question,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        add_special_tokens=False
                    )

                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    outputs = self(input_ids, mask=attention_mask) 

                    labels_tensor = torch.tensor(batch_labels).to(self.device) 

                    loss = self.criterion(outputs, labels_tensor)
                    total_loss += loss
                    num_batches += 1

                    # print("Train labels_tensor:", labels_tensor)
                    # print("Train outputs:", outputs)


                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    
                wandb.log({"train_loss": total_loss / num_batches}, step=epoch)
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
    project="modernbert-project-experimental",
    # Track hyperparameters and run metadata.
    config=args,
    )

    main(args)