import json

from datasets import Dataset
from haystack.components.readers import ExtractiveReader
from loguru import logger
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    BertForQuestionAnswering, BertConfig, BertTokenizer, AdamW

def load_json_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_datasets(train_file, val_file):
    # Load the dataset from JSON or JSONL file
    train_data = load_json_dataset(train_file)
    val_data = load_json_dataset(val_file)

    # Convert the data to Dataset format
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)

    print(train_dataset)

def train_baseline_transformer(train_file, val_file=None, num_epochs=10):
    logger.debug("Training baseline")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig()
    model = BertForQuestionAnswering(config=config)
    logger.debug("Done instantiating model")

    # Define Training Parameters
    training_args = TrainingArguments(
        output_dir="./base-results",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_file,
        eval_dataset=val_file,
        tokenizer=tokenizer
        #optimizers=AdamW(params=model.parameters(), lr=1e-3)
    )
    trainer.train()

    # Evaluation
    trainer.evaluate()

    # Save the Model
    model.save_pretrained("baseline")
    logger.debug("training baseline done")


def train_qa_transformer(train_file, val_file=None, num_epochs=10, base="deepset/roberta-base-squad2-distilled"):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForQuestionAnswering.from_pretrained(base)

    # Define Training Parameters
    training_args = TrainingArguments(
        output_dir="./finetuned_results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_file,
        eval_dataset=val_file,
        tokenizer=tokenizer,
        optimizers=(AdamW(params=model.parameters(), lr=1e-3), None)
    )
    trainer.train()

    # Evaluation
    trainer.evaluate()

    # Save the Model
    model.save_pretrained("fine_tuned_roberta_base_squad2")

def fine_tuned_reader():
    reader = ExtractiveReader()

    reader.warm_up()
    return reader
