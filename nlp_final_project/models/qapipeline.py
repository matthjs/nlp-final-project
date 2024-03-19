from haystack import Pipeline
from haystack.components.builders import PromptBuilder


class QAPipeline:
    def __init__(self,
                 text_embedder,
                 retriever,
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
            self.text_embedder = None
            self.retriever = None
            self.prompt_builder = None
            self.generator = None

        def build(self):
            return QAPipeline(self.text_embedder, self.retriever, self.prompt_builder, self.generator)
