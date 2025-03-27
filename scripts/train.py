from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding 
import evaluate
import numpy as np

raw_dataset = load_dataset("csv", data_files="data/dataset.csv")["train"]

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
train_dataset = train_dataset.map(encode_labels)
train_dataset = train_dataset.remove_columns(["text", "label"])

eval_dataset = eval_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(encode_labels)
eval_dataset = eval_dataset.remove_columns(["text", "label"])

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,  # у тебя 3 класса: simple, medium, advanced
    id2label={v: k for k, v in label2id.items()},
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="model_output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="logs",
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="tensorboard"
)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

trainer.save_model("final_model")
tokenizer.save_pretrained("final_model")
