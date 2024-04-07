import torch
import torch.nn as nn
import sys

from loguru import logger

from nlp_final_project.models.reader import train_transformer, eval_transformer, hyperparam_tuning, test_transformer
from nlp_final_project.util.inference import inference


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


def main():
    # Check if the correct number of arguments are provided
    if len(sys.argv) < 2:
        logger.info("Usage: python script.py <command>")
        return

    # Extract the command-line argument
    command = sys.argv[1]

    # Perform actions based on the command
    if command == "inference":
        logger.info("Performing inference...")
        if len(sys.argv) != 3:
            inference()
        else:
            # Call function to perform inference
            inference(sys.argv[2])
    elif command == "test":
        logger.info("Running tests...")
        test_transformer()
    elif command == "train":
        # logger.info("Training model... (baseline)")
        # train_transformer()
        logger.info("Training model...")
        train_transformer(model_str="deepset/roberta-base-squad2-distilled")
    elif command == "eval":
        logger.info("Evaluating models...")
        eval_transformer()
    else:
        logger.info("Invalid command. Please use 'inference', 'test', 'train', or 'eval'.")


if __name__ == "__main__":
    main()
