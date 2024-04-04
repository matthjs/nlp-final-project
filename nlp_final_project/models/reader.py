import json

import pandas as pd
from datasets import Dataset, load_dataset, load_metric
from evaluate.visualization import radar_plot
from haystack.components.readers import ExtractiveReader
from loguru import logger
from matplotlib import pyplot as plt
from torch.optim import Adam
from evaluate import load, evaluator, QuestionAnsweringEvaluator
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    BertForQuestionAnswering, BertConfig, BertTokenizer, AdamW, DefaultDataCollator, training_args, TrainerCallback

"""
WARNING: THIS IS A FUCKING MESS AND NEEDS TO BE CLEANED UP
"""


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

class PreProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        text_answers = examples["answers"]
        examples["answers"] = []
        labels = examples["labels"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            label = labels[i]
            start_char = label[0]["start"][0]
            end_char = label[0]["end"][0]
            sequence_ids = inputs.sequence_ids(i)

            # This stupid hack is needed because a lot of automatic evaluation in
            # huggingface for question answering tasks assumes squad like format.
            # Generate predictions in the required format
            examples["answers"].append({"text": text_answers[i], "answer_start": [start_char]})

            # Generate predictions in the required format
            # prediction_text = text_answers[i] if label else ""
            # no_answer_probability = 0.0 if not label else 1.0

            # examples["answers"].append({
            #     "id": examples["key"][i],
            #     "prediction_text": prediction_text,
            #     "no_answer_probability": no_answer_probability
            # })

            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs


def train_transformer(dataset_str="lucadiliello/newsqa", model_str="distilbert/distilbert-base-uncased", save=True):
    # Load the dataset
    dataset = load_dataset(dataset_str)

    train_set = dataset["train"] #.select(range(100))
    validation_set = dataset["validation"] #select(range(100))

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    preprocessor = PreProcessor(tokenizer)

    tokenized_train = train_set.map(preprocessor.preprocess_function, batched=True)
    tokenized_val = validation_set.map(preprocessor.preprocess_function, batched=True)

    model = AutoModelForQuestionAnswering.from_pretrained(model_str)
    data_collator = DefaultDataCollator()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_str,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
        logging_dir="./logs/" + model_str,  # Specify the directory for TensorBoard logs
        report_to=["tensorboard"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    train_results = trainer.train()

    # Evaluate the model
    evaluate_results = trainer.evaluate()

    if save:
        trainer.save_model(model_str)

    return tokenized_val

def eval_transformer(dataset_str="lucadiliello/newsqa", models_strs=["distilbert/distilbert-base-uncased_trained", "deepset/roberta-base-squad2-distilled_trained"]):
    dataset = load_dataset(dataset_str)    # This is a bit eh.
    validation_set = dataset["validation"]

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(models_strs[0])
    preprocessor = PreProcessor(tokenizer)
    tokenized_val = validation_set.map(preprocessor.preprocess_function, batched=True)

    task_evaluator: QuestionAnsweringEvaluator = evaluator("question-answering")

    results = []
    for model_str in models_strs:
        results.append(task_evaluator.compute(
            model_or_pipeline=model_str,
            data=tokenized_val,
            metric="squad_v2",
            # strategy="bootstrap", # Expensive
            id_column="key",
            # n_resamples=3,
            squad_v2_format=True
        ))
    df = pd.DataFrame(results, index=models_strs)

    print(results)

    plot = radar_plot(data=results, model_names=models_strs, invert_range=["latency_in_seconds"])
    plot.show()

def fine_tuned_reader():
    reader = ExtractiveReader()

    reader.warm_up()
    return reader
