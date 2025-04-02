from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import pandas as pd
import torch
import os

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train():
    # Load data
    train_df = pd.read_csv("../data/processed/train.csv")
    test_df = pd.read_csv("../data/processed/test.csv")
    
    # Tokenize
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_df["cleaned_prompt"].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_df["cleaned_prompt"].tolist(), truncation=True, padding=True)
    
    # Create datasets
    train_dataset = PromptDataset(train_encodings, train_df["target"].tolist())
    test_dataset = PromptDataset(test_encodings, test_df["target"].tolist())
    
    # Train
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "benign", 1: "malicious"},
        label2id={"benign": 0, "malicious": 1}
    )
    
    training_args = TrainingArguments(
        output_dir="../models/training_outputs",
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    trainer.train()
    model.save_pretrained("../models/model")
    tokenizer.save_pretrained("../models/tokenizer")

if __name__ == "__main__":
    train()