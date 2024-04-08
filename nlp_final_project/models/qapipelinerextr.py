import os
import pickle
from typing import Tuple

from haystack import Pipeline, Document, ExtractedAnswer
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from loguru import logger

from nlp_final_project.models.qapipeline import QAPipeline


def simplify_prediction_output(prediction: dict, default_no_answer_str="<no answer>") -> str:
    """
    Simplify the prediction output into a human-readable string.

    :param default_no_answer_str:
    :param prediction: The prediction dictionary containing answers.
    :return: Human-readable string containing question, confidence, and answer.
    """
    return "\n".join([f"Question: {answer.query}\nConfidence: {round(answer.score, 2)}\nAnswer: "
                      f"{answer.data if answer.data is not None else default_no_answer_str}\n" \
                      for answer in prediction['answers']])


class QAPipelineRetrieverExtractor(QAPipeline):
    """
    Retriever Extractor QA system.
    """

    def __init__(self,
                 documents: list[Document],
                 doc_embedder: SentenceTransformersDocumentEmbedder,
                 document_store: DocumentStore,
                 text_embedder: SentenceTransformersTextEmbedder,
                 retriever: ExtractiveReader,
                 reader: ExtractiveReader,
                 index_documents=True):
        """
        Initialize the QAPipelineRetrieverExtractor.

        :param documents: List of documents to be indexed.
        :param doc_embedder: Document embedder instance.
        :param document_store: Document store to index documents.
        :param text_embedder: Text embedder instance.
        :param retriever: Retriever instance.
        :param reader: Extractive reader instance.
        :param index_documents: Whether to index documents on initialization. Defaults to True.
        """
        # indexing pipeline fetches documents, processes it and loads into document store.
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("embedder", doc_embedder)
        # DocumentWriter writes the vectorized documents to the DocumentStore.
        self.indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
        self.indexing_pipeline.connect("embedder.documents", "writer.documents")

        if index_documents:  # You do not want to do this if you load the document_store object.
            logger.debug("Running indexing pipeline on documents...")
            self.indexing_pipeline.run({"documents": documents})
            logger.debug("Done running index pipeline on documents!")

            with open("emb.pkl", "wb") as pickle_file:
                pickle.dump(document_store, pickle_file)

        # Extractive QA pipeline consists of an embedder, retriever, and reader.
        self.extractive_qa_pipeline = Pipeline()
        self.extractive_qa_pipeline.add_component(instance=text_embedder, name="embedder")
        self.extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
        self.extractive_qa_pipeline.add_component(instance=reader, name="reader")

        # Now, connect the components to each other
        self.extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.extractive_qa_pipeline.connect("retriever.documents", "reader.documents")

    def answer_question(self, query: str, context_string: str = None) -> Tuple[str, dict]:
        """
        Answer a question either with or without context string.

        :param query: The question to be answered.
        :param context_string: The context string (optional).
        :return: Answer to the question and prediction dictionary.
        """
        if context_string is not None:
            return self.answer_question_with_context(query, context_string)

        logger.debug("No context_string provided, performing information retrieval on stored documents...")

        prediction = self.extractive_qa_pipeline.run({"embedder": {"text": query}, "reader": {"query": query}})
        return simplify_prediction_output(prediction), prediction

    def answer_question_with_context(self, query: str, context_string: str) -> Tuple[str, dict]:
        """
        Answer a question with provided context string.

        :param query: The question to be answered.
        :param context_string: The context string for answering the question.
        :return: Answer to the question and prediction dictionary.
        """

        # Run reader on the query and the context string
        prediction = self.extractive_qa_pipeline.get_component("reader").run(
            query=query,
            documents=[Document(content=context_string)])

        return simplify_prediction_output(prediction), prediction

    class QABuilder:
        """
        Helper class to build QAPipelineRetrieverExtractor.
        """

        def __init__(self):
            # Retrieving part.
            # DocumentStore is use to index retrieved documents
            self.document_store = None

            # Transforms each document into a vector
            self.embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
            self.doc_embedder = SentenceTransformersDocumentEmbedder(model=self.embedding_model)
            self.doc_embedder.warm_up()

            # Reading part.
            # Creates an embedding for user query.
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.embedding_model)
            self.text_embedder.warm_up()
            # This will get the relevant documents to the query.
            self.retriever = None

            # The ExtractiveReader returns answers to that query,
            # as well as their location in the source document, and a confidence score.
            self.reader = None

            self.docs = None  # Has to be set
            self.index_documents = True

        def set_docs(self, docs: list[Document]) -> 'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set documents for indexing.

            :param docs: List of documents.
            :return: Builder instance.
            """
            self.docs = docs
            return self

        def set_docs_embedder(self, doc_embedder: SentenceTransformersDocumentEmbedder) -> \
                'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set document embedder.

            :param doc_embedder: Document embedder instance.
            :return: Builder instance.
            """
            doc_embedder.warm_up()
            self.doc_embedder = doc_embedder
            return self

        def set_text_embedder(self, text_embedder: SentenceTransformersTextEmbedder) -> \
                'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set text embedder.

            :param text_embedder: Text embedder instance.
            :return: Builder instance.
            """
            self.text_embedder = text_embedder
            return self

        def set_document_store(self, document_store: DocumentStore) -> 'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set document store.

            :param document_store: Document store instance.
            :return: Builder instance.
            """
            self.document_store = document_store
            return self

        def set_retriever(self, retriever: ExtractiveReader) -> 'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set retriever.

            :param retriever: Retriever instance.
            :return: Builder instance.
            """
            self.retriever = retriever
            return self

        def set_reader(self, reader: ExtractiveReader) -> 'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set reader.

            :param reader: Extractive reader instance.
            :return: Builder instance.
            """
            self.reader = reader
            return self

        def set_index_documents(self, index_documents: bool) -> 'QAPipelineRetrieverExtractor.QABuilder':
            """
            Set whether to index documents.

            :param index_documents: Whether to index documents.
            :return: Builder instance.
            """
            self.index_documents = index_documents
            return self

        def build(self) -> 'QAPipelineRetrieverExtractor':
            """
            Build QAPipelineRetrieverExtractor instance.
            :return: QAPipelineRetrieverExtractor.
            """
            if self.index_documents is True:
                if self.document_store is None and self.retriever is None:
                    raise ValueError("Retriever or document store not set")

                if self.docs is None:
                    raise ValueError("No documents set")

            logger.debug("Running QAPipeline constructor.")

            return QAPipelineRetrieverExtractor(documents=self.docs,
                                                doc_embedder=self.doc_embedder,
                                                document_store=self.document_store,
                                                text_embedder=self.text_embedder,
                                                retriever=self.retriever,
                                                reader=self.reader,
                                                index_documents=self.index_documents)


def elastic_search_retriever(host=os.environ.get("ELASTICSEARCH_HOST", "https://localhost:9200")):
    # Tricky to setup locally, should probably use with docker or something.
    document_store = ElasticsearchDocumentStore(hosts=QAPipelineRetrieverExtractor.QABuilder.host,
                                                verify_certs=False,  # This in insecure
                                                basic_auth=("elastic", ""))
    retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)
    return document_store, retriever


def in_memory_retriever(load=False) -> tuple:
    """
    Retrieve documents using in-memory storage.

    :param load: Whether to load from file. Defaults to False.
    :return: Tuple containing whether loaded from file, DocumentStore, and retriever instances.
    """
    # Simpler, but less powerful storage for document embeddings.
    logger.debug("Instantiating document store")
    try:
        if load:
            with open("emb.pkl", "rb") as f:
                document_store = pickle.load(f)
            retriever = InMemoryEmbeddingRetriever(document_store=document_store)
            return True, document_store, retriever
    except FileNotFoundError:
        logger.warning("File 'emb.pkl' not found.")
        pass

    document_store = InMemoryDocumentStore()
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    return False, document_store, retriever
