from datasets import load_dataset
from haystack import Document
from loguru import logger

from nlp_final_project.models.qapipelinerextr import in_memory_retriever, QAPipelineRetrieverExtractor
from nlp_final_project.models.reader import fine_tuned_reader


def build_qa_pipeline_wo_documents():
    """
    Build a Question Answering (QA) pipeline without providing documents.

    :return: Built QA pipeline.
    """
    succesfully_loaded, document_store, retriever = in_memory_retriever(
        load=True)  # Local storage for document embeddings and associated retrie

    return (QAPipelineRetrieverExtractor.QABuilder()
            .set_document_store(document_store)
            .set_retriever(retriever)
            .set_reader(fine_tuned_reader())
            .set_index_documents(False)
            .build())


def build_qa_pipeline(documents: str = None):
    """
    Build a Question Answering (QA) pipeline using provided documents.

    :param documents: Path to the dataset for inference.
    :return: Built QA pipeline.
    """
    if documents is None:
        return build_qa_pipeline_wo_documents()

    # Load the dataset for inference.
    dataset = load_dataset(documents)

    train_set = dataset["train"]
    validation_set = dataset["validation"]

    docs_t = [Document(content=doc["context"], id=doc["key"]) for doc in train_set]
    docs_v = [Document(content=doc["context"], id=doc["key"]) for doc in validation_set]
    docs = docs_t + docs_v

    logger.debug("Done loading dataset")

    succesfully_loaded, document_store, retriever = in_memory_retriever(
        load=True)  # Local storage for document embeddings and associated retriever.

    return (QAPipelineRetrieverExtractor.QABuilder()
            .set_docs(docs_v)
            .set_document_store(document_store)
            .set_retriever(retriever)
            .set_reader(fine_tuned_reader())
            .set_index_documents(not succesfully_loaded)
            .build())


def inference(documents: str = None):
    """
    Run the inference process for the QA system.

    :param documents: Path to the dataset for inference. Defaults to None.
    """
    logger.info("Building QA Pipeline...")
    qa_pipeline = build_qa_pipeline(documents)
    logger.info("QA Pipeline Built Successfully!")

    logger.info("Welcome to the QA System. Type 'exit' to quit.")
    while True:
        user_input = input("Ask a question (format: <question string> : <context>): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Split input into question and context
        question, _, context = user_input.partition(" : ")
        if not context:
            answer, _ = qa_pipeline.answer_question(question)
        else:
            answer, _ = qa_pipeline.answer_question(question, context_string=context)
        print(answer)
