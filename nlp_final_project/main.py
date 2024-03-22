# Check if pytorch works as expected
import torch
import torch.nn as nn
from haystack import Document
from datasets import load_dataset
from loguru import logger

from nlp_final_project.models.qapipeline import QAPipeline


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
    # Random dataset.
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

    logger.debug("Done loading dataset")
    qa = QAPipeline.QABuilder().set_docs(docs).build()
    qa.answer_question("What does Rhodes Statue look like?")

