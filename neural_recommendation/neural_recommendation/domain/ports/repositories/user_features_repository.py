from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class UserFeaturesRepository(ABC):
    @abstractmethod
    def get_features(self, user_id: int) -> Optional[torch.Tensor]:
        pass

    @abstractmethod
    def get_similar_users(self, user_features: torch.Tensor, top_k: int) -> List[int]:
        pass
