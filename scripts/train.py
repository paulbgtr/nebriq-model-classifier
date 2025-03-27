from datasets import load_dataset
from transformers import AutoTokenizer

raw_dataset = load_dataset("csv", data_files="../data/dataset.csv")["train"]

split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

label2id = {"simple": 0, "medium": 1, "advanced": 2}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=128)

def encode_labels(batch):
    batch["labels"] = label2id[batch["label"]] 
    return batch

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)
