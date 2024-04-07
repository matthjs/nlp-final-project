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
from matplotlib import pyplot as plt
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
from datasets import disable_caching

from nlp_final_project.models.preprocessor import PreProcessor


def fixed_grid_search(dataset_str="lucadiliello/newsqa", model_str="deepset/roberta-base-squad2-distilled",
                      data_size=4000):
    """
    Perform a fixed grid search to find the best number of epochs for training a Question Answering model.

    :param dataset_str: Name of the dataset for training. Defaults to "lucadiliello/newsqa".
    :param model_str: Pretrained model to be fine-tuned. Defaults to "deepset/roberta-base-squad2-distilled".
    :param data_size: Maximum size of the (train) data to use for training. Defaults to 4000.
    """
    num_epochs = [1, 2, 3, 5, 7]

    f1_scores_run = []

    for num_train_epochs in num_epochs:
        train_transformer(dataset_str=dataset_str,
                          model_str=model_str,
                          max_data_size=data_size,
                          save=True,
                          num_train_epochs=num_train_epochs)

        f1_scores, em_scores = eval_transformer(dataset_str=dataset_str,
                                                models_strs=np.array([model_str]),
                                                max_data_size=data_size)
        # This is stupid
        shutil.rmtree(model_str)
        shutil.rmtree("./logs/" + model_str)
        f1_scores_run.append(f1_scores[0])

    # Find the best F1 score and corresponding number of epochs
    best_f1_score = max(f1_scores_run)
    best_num_epochs = num_epochs[f1_scores_run.index(best_f1_score)]

    print("Best number of epochs:", best_num_epochs)
    print("Best F1 score:", best_f1_score)

    print(num_epochs)
    print(f1_scores_run)

    # Plot the results
    plt.plot(num_epochs, f1_scores_run, marker='o')
    plt.title("F1 Score vs Number of Epochs")
    plt.xlabel("Number of Epochs")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.savefig("f1_score_vs_epochs.png")

    # Save the results to a CSV file
    results_dict = {"Number of Epochs": num_epochs, "F1 Score": f1_scores_run}
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv("f1_score_vs_epochs.csv", index=False)


def hyperparam_tuning(dataset_str="lucadiliello/newsqa", model_str="deepset/roberta-base-squad2-distilled",
                      data_size=4000):
    """
    Perform hyperparameter tuning for a Question Answering model.
    This is *very* computationally expensive. Only run this if you have a big enough gear and a lot of time.

    :param dataset_str: Name of the dataset for tuning. Defaults to "lucadiliello/newsqa".
    :param model_str: Pretrained model to be fine-tuned. Defaults to "deepset/roberta-base-squad2-distilled".
    :param data_size: Maximum size of the (train) data to use for tuning. Defaults to 4000.
    """
    disable_caching()
    lock = threading.Lock()  # Thread safety.

    def objective(trial):
        """


        :param trial:
        :return: F1 score on the evaluation set (silently assumed to be a comparable size to the train set.
        """
        # learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)    Not really necessary.
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
                          max_data_size=data_size)

        f1_scores, _ = eval_transformer(dataset_str, models_strs=np.array([model_str]), display=False,
                                        max_length=max_length)
        with lock:
            # This is stupid
            shutil.rmtree(model_str)
            shutil.rmtree("./logs/" + model_str)
        return f1_scores[0]

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=20)  # You can adjust the number of trials

    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    # Save best hyperparameters to a file
    with open("best_hyperparameters.json", "w") as file:
        json.dump(best_params, file)


def train_transformer(dataset_str="lucadiliello/newsqa", model_str="distilbert/distilbert-base-uncased", save=True,
                      max_length=512, learning_rate=2e-5, batch_size=16, num_train_epochs=3, max_data_size=None):
    """
    Train (fine tune) a Question Answering BERT model.

    :param dataset_str: Name of the dataset for training.
    :param model_str: Pretrained model to be fine-tuned.
    :param save: Whether to save the trained model. Defaults to True.
    :param max_length: Maximum length of input sequences. Defaults to 512.
    :param learning_rate: Learning rate for training. Defaults to 2e-5.
    :param batch_size: Batch size for training. Defaults to 16.
    :param num_train_epochs: Number of training epochs. Defaults to 3.
    :param max_data_size: Maximum number of data samples to use for training. Defaults to None.
    :return: Tokenized validation dataset.
    """
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
                     models_strs=np.array(["distilbert/distilbert-base-uncased_final",
                                           "deepset/roberta-base-squad2-distilled_final",
                                           "VMware/electra-small-mrqa"]),
                     display=False,
                     max_data_size=None,
                     max_length=512):    # This is a bit eh.
    """
    Evaluate a fine-tuned Question Answering BERT model. Evaluation is done on the dev/evaluation set
    of the passed dataset.
    The function saves the results to a CSV file.

    :param dataset_str: Name of the dataset for evaluation.
    :param models_strs: List of pre-trained model strings.
    :param display: Whether to display evaluation results. Defaults to True.
    :param max_data_size: Maximum number of data samples to use for evaluation. Defaults to None.
    :param max_length: Maximum length of input sequences. Defaults to 512.
    :return: F1 scores and exact match scores.
    """
    dataset = load_dataset(dataset_str)

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
    # df = pd.DataFrame(results, index=models_strs)
    # df.to_csv(dataset_str + ".csv")

    print(results)

    if display:
        print(results)

        # This is broken.
        # plot = radar_plot(data=results, model_names=models_strs, invert_range=["latency_in_seconds"])
        # plot.show()

    return [entry['f1'] for entry in results], [entry['exact'] for entry in results]


def test_transformer(models_strs=np.array(["distilbert/distilbert-base-uncased_final",
                                           "deepset/roberta-base-squad2-distilled_final",
                                           "deepset/roberta-base-squad2-distilled_trained",
                                           "VMware/electra-small-mrqa"])):
    """
    Evaluate pre-trained Question Answering BERT models on various datasets.
    The datasets is a selection from the test set of (https://huggingface.co/datasets/mrqa).
    These are all out of distribution datasets.
    - TextbookQA (1503 samples)
    - DROP (1503 samples)
    - BioASQ (1504 samples)
    - RACE (674 samples)
    - DuoRC (1501 samples)
    NOTE: DO NOT RUN THIS FUNCTION UNTIL AFTER FINE TUNING.
    """
    eval_transformer(dataset_str="lucadiliello/textbookqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/dropqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/bioasqqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/raceqa", models_strs=models_strs)
    eval_transformer(dataset_str="lucadiliello/duorc.paraphrasercqa", models_strs=models_strs)


def fine_tuned_reader(model_str="deepset/roberta-base-squad2-distilled_final"):
    """
    Initialize a fine-tuned Question Answering BERT reader.

    :param model_str: Pretrained model to be fine-tuned.
    :return: Initialized fine-tuned reader to be used in a QA pipeline object.
    """
    reader = ExtractiveReader(model_str)

    reader.warm_up()
    return reader
