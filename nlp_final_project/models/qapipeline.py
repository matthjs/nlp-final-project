from abc import ABC, abstractmethod


class QAPipeline(ABC):
    @abstractmethod
    def answer_question(self, query: str) -> str:
        pass
