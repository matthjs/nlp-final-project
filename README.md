<br />
<p align="center">
  <h1 align="center">Natural Language Processing - Final Project</h1>
    
  <p align="center">
  </p>
</p>

## About The Project
This project is about Retriever-Extractor based Question Answering.

## Getting started

### Prerequisites
- [Docker v4.25](https://www.docker.com/get-started) or higher (if running docker container).
- [Poetry](https://python-poetry.org/).
## Running
Using docker: Run the docker-compose files to run all relevant services (`docker compose up` or `docker compose up --build`).

You can also set up a virtual environment using Poetry. Poetry can  be installed using `pip`:
```
pip install poetry
```
Then initiate the virtual environment with the required dependencies (see `poetry.lock`, `pyproject.toml`):
```
poetry config virtualenvs.in-project true    # ensures virtual environment is in project
poetry install
```
The virtual environment can be accessed from the shell using:
```
poetry shell
```
IDEs like Pycharm will be able to detect the interpreter of this virtual environment.

## Models

All the Transformer models used in this project can be downloaded from HuggingFace (this will be automatically done by the code).
* [Sentence-transformer for semantic search](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
* [Baseline Extractive Reader](https://huggingface.co/Matthijs0/DistilBERT)
* [Extractive Reader](https://huggingface.co/Matthijs0/Distilled-RoBERTa)

## Usage

The project is run through the `main.py` script. You can give it the following command line arguments:
1. **Perform Inference:**
   ```bash
   python script.py inference
   ```
   To perform inference on a specific document collection:
   ```bash
   python script.py inference <hugging_face_dataset>
   ```

2. **Train [baseline](https://huggingface.co/distilbert/distilbert-base-uncased) reader model and [primary](https://huggingface.co/deepset/roberta-base-squad2-distilled) reader model on NewsQA train set:**
   ```bash
   python script.py train
   ```

3**Evaluate reader models (baseline, primary) on NewsQA evaluation set:**
   ```bash
   python script.py eval
   ```

4**Evaluate reader models (baseline, primary) on out of distribution test sets:**
   ```bash
   python script.py test
   ```



