import json
import os
import shutil
import threading

import numpy as np
import optuna
import pandas as pd
from datasets import load_dataset
from evaluate.visualization import radar_plot
from haystack.components.readers import ExtractiveReader
from evaluate import evaluator, QuestionAnsweringEvaluator
from optuna.multi_objective.trial import MultiObjectiveTrial
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import disable_caching

"""

"""


class PreProcessor:
    def __init__(self, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_function(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
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


def hyperparam_tuning(dataset_str="lucadiliello/newsqa", model_str="deepset/roberta-base-squad2-distilled",
                      data_size=4000):
    disable_caching()
    lock = threading.Lock()

    def objective(trial):
        # learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
        # With 80 I run out of memory on my laptop GPU.
        per_device_batch_size = trial.suggest_categorical("per_device_batch_size", [16, 32, 64])
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 5)
        max_length = trial.suggest_int("max_length", low=384, high=512, step=128)

        train_transformer(dataset_str=dataset_str,
                          model_str=model_str,
                          save=True,
                          max_length=max_length,
                          batch_size=per_device_batch_size,
                          num_train_epochs=num_train_epochs,
                          max_data_size=data_size,
                          force_download=True)

        f1_scores = eval_transformer(dataset_str, models_strs=np.array([model_str]), display=False,
                                     max_length=max_length)
        with lock:
            shutil.rmtree(model_str)
            shutil.rmtree("./logs/" + model_str)  # This is stupid
        return f1_scores[0]

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=20)  # You can adjust the number of trials

    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    # Save best hyperparameters to a file
    with open("best_hyperparameters.json", "w") as file:
        json.dump(best_params, file)


def train_transformer(dataset_str="lucadiliello/newsqa", model_str="distilbert/distilbert-base-uncased", save=True,
                      max_length=512, learning_rate=2e-5, batch_size=16, num_train_epochs=3, max_data_size=None,
                      force_download=False):
    # Load the dataset
    dataset = load_dataset(dataset_str)

    if max_data_size is None:
        train_set = dataset["train"]
        validation_set = dataset["validation"]
    else:
        train_set = dataset["train"].select(range(max_data_size))
        validation_set = dataset["validation"].select(range(max_data_size))

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    preprocessor = PreProcessor(tokenizer, max_length)

    tokenized_train = train_set.map(preprocessor.preprocess_function, batched=True)
    tokenized_val = validation_set.map(preprocessor.preprocess_function, batched=True)

    model = AutoModelForQuestionAnswering.from_pretrained(model_str)
    data_collator = DefaultDataCollator()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_str,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
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


def eval_transformer(dataset_str="lucadiliello/newsqa",
                     models_strs=np.array(["distilbert/distilbert-base-uncased_trained",
                                           "deepset/roberta-base-squad2-distilled_trained",
                                           "VMware/electra-small-mrqa"]),
                     display=True,
                     max_data_size=None,
                     max_length=384):
    dataset = load_dataset(dataset_str)  # This is a bit eh.

    if max_data_size is None:
        validation_set = dataset["validation"]
    else:
        validation_set = dataset["validation"].select(range(max_data_size))

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(models_strs[0])
    preprocessor = PreProcessor(tokenizer, max_length)
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
    df.to_csv(dataset_str + ".csv")

    if display:
        print(results)

        # This is broken.
        # plot = radar_plot(data=results, model_names=models_strs, invert_range=["latency_in_seconds"])
        # plot.show()

    return [entry['f1'] for entry in results]


def test_transformers(models_strs=np.array(["distilbert/distilbert-base-uncased_trained",
                                            "deepset/roberta-base-squad2-distilled_trained",
                                            "VMware/electra-small-mrqa"])):
    """
    DO NOT TOUCH THIS UNTIL THE END OF TUNING.
    :param models_strs:
    :return:
    """
    eval_transformer(dataset_str="lucadiliello/textbookqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/dropqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/bioasqqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/raceqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/duorc.paraphrasercqa", models_strs=models_strs)


def fine_tuned_reader():
    reader = ExtractiveReader("deepset/roberta-base-squad2-distilled_trained")

    reader.warm_up()
    return reader
