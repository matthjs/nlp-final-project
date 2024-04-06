from abc import ABC, abstractmethod
from typing import Tuple


class QAPipeline(ABC):
    """
    Abstract base class for Question Answering Pipelines.
    """
    @abstractmethod
    def answer_question(self, query: str, context_string: str = None) -> Tuple[str, dict]:
        pass
