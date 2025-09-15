from abc import ABC, abstractmethod
from typing import Any, Dict


class LLMPort(ABC):
    @abstractmethod
    def extract_user_profile(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def chat(self, system: str, user: str) -> str:
        pass

    @abstractmethod
    def extract_liked_movie_ids(self, movies: Dict[str, Any], user_reply: str) -> Dict[str, Any]:
        pass
