import json

import torch
import torch.nn as nn
from haystack import Document
from datasets import load_dataset
from loguru import logger

from nlp_final_project.models.qapipelinerextr import QAPipelineRetrieverExtractor, in_memory_retriever
from nlp_final_project.models.reader import train_qa_transformer


def gpu_test():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Define a simple neural network and move it to the GPU
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1)).to(device)

    # Example of using the model
    input_data = torch.randn(1, 10).to(device)
    output = model(input_data)
    print("Output:", output.item())


if __name__ == "__main__":
    train_qa_transformer(train_file="../data/train-v2.0.json", val_file="../data/dev-v2.0.json")
    # Load dataset
    #with open('../data/train-v2.0.json', 'r') as json_file:
    #    train_set = json.load(json_file)

    #print(train_set)
    """
    dataset = load_dataset("squad")
    print(dataset)
    train_set = dataset["train"]
    validation_set = dataset["validation"]
    docs = [Document(content=doc["context"], id=doc["id"]) for doc in train_set]
    docs = docs[0:5]

    logger.debug("Done loading dataset")

    document_store, retriever = in_memory_retriever(load=False)  # Local storage for document embeddings and associated retriever.

    qa = (QAPipelineRetrieverExtractor.QABuilder()
          .set_docs(docs)
          .set_document_store(document_store)
          .set_retriever(retriever)
          .set_index_documents(True)
          .build())

    logger.debug("Done constructing pipeline")
    print(qa.answer_question("What was Beyonc\u00e9's first acting job, in 2006?"))
    """
