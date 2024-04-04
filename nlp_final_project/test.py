import torch
from accelerate import DataLoaderConfiguration, Accelerator
from transformers import BertForQuestionAnswering, BertTokenizer, TrainingArguments, Trainer, AutoTokenizer, \
    AutoModelForQuestionAnswering, DefaultDataCollator
from datasets import load_dataset

def main():
    # Load the SQuAD dataset
    dataset = load_dataset("lucadiliello/newsqa")

    train_set = dataset["train"]
    validation_set = dataset["validation"]

    # Load pre-trained BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased_trained")

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        labels = examples["labels"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            label = labels[i]
            start_char = label[0]["start"][0]
            end_char = label[0]["end"][0]
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
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

    tokenized_train = train_set.map(preprocess_function, batched=True)
    tokenized_val = validation_set.map(preprocess_function, batched=True)

    model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased_trained")
    data_collator = DefaultDataCollator()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="my_awesome_qa_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=False,
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
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model
    trainer.save_model("base_model")

if __name__ == "__main__":
    main()
