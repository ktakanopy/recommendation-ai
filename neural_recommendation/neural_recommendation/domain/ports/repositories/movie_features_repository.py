import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class MovieFeaturesRepository(ABC):
    @abstractmethod
    def get_features(self, movie_id: int) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def get_similar_movies(self, movie_id: int, top_k: int) -> List[int]:
        pass

    @abstractmethod
    def get_top_popular_movies(self, top_k: int) -> List[int]:
        pass
