

import os
from typing import Any, Dict

import torch

from two_towers.domain.models.deep_learning.model_config import ModelConfig
from two_towers.domain.models.deep_learning.two_tower_model import TwoTowerModel


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path

    @staticmethod
    def load_model(config: ModelConfig, additional_feature_info: Dict[str, Any]) -> TwoTowerModel:

        model_path = os.path.join(config.models_dir, config.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = TwoTowerModel(
            layer_sizes=config.layer_sizes,
            unique_user_ids=additional_feature_info["unique_user_ids"],
            embedding_size=config.embedding_size,
            additional_feature_info=additional_feature_info,
            device=config.device,
        )

        state_dict = torch.load(model_path, map_location=config.device)

        model_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(config.device)

        return model
