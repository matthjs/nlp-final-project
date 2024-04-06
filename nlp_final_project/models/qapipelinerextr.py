import os
import pickle

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


def simplify_prediction_output(prediction: dict) -> str:
    return "\n".join([f"Question: {answer.query}\nConfidence: {answer.score}\nAnswer: {answer.data}\n" \
                      for answer in prediction['answers'] if answer.data is not None])


class QAPipelineRetrieverExtractor(QAPipeline):
    """
    Retriever Extractor QA system.
    https://haystack.deepset.ai/tutorials/34_extractive_qa_pipeline
    Note that there might actually be some prebuilt pipelines (ExtractiveQAPipeline).
    """

    def __init__(self,
                 documents,
                 doc_embedder: SentenceTransformersDocumentEmbedder,
                 document_store: DocumentStore,
                 text_embedder: SentenceTransformersTextEmbedder,
                 retriever,
                 reader: ExtractiveReader,
                 index_documents=True):
        # indexing pipeline fetches documents, processes it and loads into document store.
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("embedder", doc_embedder)
        # DocumentWriter writes the vectorized documents to the DocumentStore.
        self.indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
        self.indexing_pipeline.connect("embedder.documents", "writer.documents")

        if index_documents:    # You do not want to do this if you load the document_store object.
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

    def answer_question(self, query: str, context_string: str = None) -> tuple:
        if context_string is not None:
            return self.answer_question_with_context(query, context_string)

        logger.debug("No context_string provided, performing information retrieval on stored documents...")

        prediction = self.extractive_qa_pipeline.run({"embedder": {"text": query}, "reader": {"query": query}})
        return simplify_prediction_output(prediction), prediction

    def answer_question_with_context(self, query: str, context_string: str) -> tuple:
        """
        Run the text embedder on the provided context string and then run the reader on the query and the context string.

        Args:
            query (str): The query for which the answer is sought.
            context_string (str): The context string to be used for answering the question.

        Returns:
            str: The answer to the given query within the provided context string.
        """
        # Run text embedder on the context string
        # context_embedding = self.extractive_qa_pipeline.get_component("embedder").run(context_string)

        # Run reader on the query and the context string
        prediction = self.extractive_qa_pipeline.get_component("reader").run(
            query=query,
            documents=[Document(content=context_string)])

        return simplify_prediction_output(prediction), prediction

    class QABuilder:
        """
        Helper class to create model.
        """

        def __init__(self):
            # Retrieving part.
            # DocumentStore is use to index retrieved documents
            self.document_store = None

            # Transforms each document into a vector
            self.model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
            self.doc_embedder = SentenceTransformersDocumentEmbedder(model=self.model)
            self.doc_embedder.warm_up()

            # Reading part.
            # Creates an embedding for user query.
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.model)
            self.text_embedder.warm_up()
            # This will get the relevant documents to the query.
            self.retriever = None

            # The ExtractiveReader returns answers to that query,
            # as well as their location in the source document, and a confidence score.
            self.reader = None

            self.docs = None  # Has to be set
            self.index_documents = True

        def set_docs(self, docs):
            self.docs = docs
            return self

        def set_docs_embedder(self, doc_embedder):
            doc_embedder.warm_up()
            self.doc_embedder = doc_embedder
            return self

        def set_text_embedder(self, text_embedder):
            self.text_embedder = text_embedder
            return self

        def set_document_store(self, document_store):
            self.document_store = document_store
            return self

        def set_retriever(self, retriever):
            self.retriever = retriever
            return self

        def set_reader(self, reader):
            self.reader = reader
            return self

        def set_index_documents(self, index_documents):
            self.index_documents = index_documents
            return self

        def build(self):
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


def in_memory_retriever(load=False):
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
