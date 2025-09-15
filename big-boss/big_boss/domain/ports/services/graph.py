from abc import ABC, abstractmethod
from typing import Any


class GraphPort(ABC):
    @abstractmethod
    async def run(self, user_id: str, message: str) -> Any:
        pass
