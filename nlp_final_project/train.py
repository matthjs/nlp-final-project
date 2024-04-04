from nlp_final_project.models.reader import train_baseline_transformer, load_datasets, \
    train_transformer, eval_transformer

if __name__ == "__main__":
    # train_transformer(model_str="deepset/roberta-base-squad2-distilled")
    eval_transformer()
