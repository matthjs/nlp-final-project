from haystack.components.readers import ExtractiveReader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer


def train_qa_transformer(train_file, val_file=None, base="deepset/roberta-base-squad2-distilled"):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForQuestionAnswering.from_pretrained(base)

    # Step 5: Define Training Parameters
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    # Step 6: Fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_file,
        eval_dataset=val_file,
        tokenizer=tokenizer,
    )
    trainer.train()

    # Step 7: Evaluation
    trainer.evaluate()

    # Step 8: Save the Model
    model.save_pretrained("fine_tuned_roberta_base_squad2")


def fine_tuned_reader():
    reader = ExtractiveReader()

    reader.warm_up()
    return reader
