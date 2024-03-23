import os
from getpass import getpass

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator, HuggingFaceLocalGenerator
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from loguru import logger

from nlp_final_project.models.qapipeline import QAPipeline


class QAPipelineRAG(QAPipeline):
    """
    Retriever Augmentation QA system.
    """

    def __init__(self,
                 text_embedder: SentenceTransformersTextEmbedder,
                 retriever: InMemoryEmbeddingRetriever,
                 prompt_builder: PromptBuilder,
                 generator):
        """
        Create an end-to-end QA system as a pipeline object. Contains
        :param text_embedder:
        :param retriever:
        :param prompt_builder:
        :param generator:
        :return:
        """
        pipeline = Pipeline()

        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", generator)

        # Now, connect the components to each other
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "llm")

        self.pipeline = pipeline

    def answer_question(self, query: str) -> str:
        response = self.pipeline.run({"text_embedder": {"text": query}, "prompt_builder": {"question": query}})
        return response["llm"]["replies"][0]

    class QABuilder:
        """
        Helper class to create model.
        """
        prompt_template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

        def __init__(self):
            self.model = "sentence-transformers/all-MiniLM-L6-v2"
            # DocumentStore is use dto index retrieved documents
            # Does not scale well to larger document collections.
            # DuplicateDocumentError: ID '561189bd3098e630a5a18883b19a6457bdc4b14d7422059f6f8a016d713fb763'
            # already exists ???
            self.document_store = InMemoryDocumentStore()
            self.doc_embedder = SentenceTransformersDocumentEmbedder(model=self.model)
            self.doc_embedder.warm_up()

            # Creates an embedding for user query.
            # Later used by Retriever to retrieve relevant documents from DocumentStore.
            self.text_embedder = SentenceTransformersTextEmbedder(model=self.model)
            # This will get the relevant documents to the query.
            self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store,
                                                        top_k=2)  # Get the 2 most relevant documents.

            self.prompt_builder = PromptBuilder(template=QAPipelineRAG.QABuilder.prompt_template)
            # The generator is the component that interacts with LLMs
            logger.debug("Instantiating generator.")

            # Using an open source language model.
            # TODO: Is this model already trained on SQuAD?
            self.generator = HuggingFaceLocalGenerator(model="google/flan-t5-large",
                                                       task="text2text-generation",
                                                       generation_kwargs={
                                                           "max_new_tokens": 100,
                                                           "temperature": 0.9,
                                                           "do_sample": True
                                                       })
            self.generator.warm_up()  # What does this mean?

            self.docs = None  # Has to be set

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

        def set_retriever(self, retriever):
            self.retriever = retriever
            return self

        def set_prompt_builder(self, prompt_builder):
            self.prompt_builder = prompt_builder
            return self

        def set_generator(self, generator):
            self.generator = generator
            return self

        def build(self):
            if self.docs is None:
                raise ValueError("No documents set")

            logger.debug("Embedding documents")
            # Embedder will create embeddings for each document and save
            # these embeddings in Document object's embedding field.
            # Then, you can write the documents to the DocumentStore.
            docs_with_embeddings = self.doc_embedder.run(self.docs)
            self.document_store.write_documents(docs_with_embeddings["documents"])

            logger.debug("Running QAPipeline constructor.")

            return QAPipelineRAG(self.text_embedder, self.retriever, self.prompt_builder, self.generator)
