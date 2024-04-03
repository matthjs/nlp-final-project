from nlp_final_project.models.reader import train_baseline_transformer, load_datasets

if __name__ == "__main__":
    load_datasets("../data/train-v2.0.json", "../data/dev-v2.0.json")
    #train_baseline_transformer(train_file="../data/nq-train-sample.jsonl",
    #                           val_file="../data/nq-dev-sample.jsonl")
