"""
This file fine tunes DeBERTa v3 on the task, as was done by the winning team
https://arxiv.org/pdf/2403.00809
""" 
import argparse

import wandb
from dataset import get_huggingface_dataset
from transformers import AutoModelForSequenceClassification, AutoConfig, DebertaTokenizer, Trainer, TrainingArguments
import numpy as np
import evaluate

def main(args):
    # load model, config, tokenizer, dataset
    MODEL_NAME = 'microsoft/deberta-v3-base'
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)
    dataset = get_huggingface_dataset( sp=True, root_dir="./data/")

    # load tokenizer, define and apply preprocessing function
    tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

    def tokenize(examples):
            # tokenize the data
            return tokenizer(
                examples["formatted_question"], 
                padding="max_length", 
                truncation=True, 
                max_length=105
                )

    dataset = dataset.map(tokenize, batched=True).train_test_split(test_size=0.1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # convert the logits to their predicted class
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir="my_model",
        eval_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

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
    project="deberta-project-local",
    # Track hyperparameters and run metadata.
    config=args,
    )

    main(args)