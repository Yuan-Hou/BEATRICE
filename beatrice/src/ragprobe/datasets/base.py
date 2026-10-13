from abc import ABC, abstractmethod
from typing import Iterable, Dict, Any


class DatasetBuilder(ABC):
    name: str

    @abstractmethod
    def build(self) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError
