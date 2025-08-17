import argparse

import torch

from two_towers.applications.use_cases.deep_learning.model_trainer import ModelTrainer
from two_towers.domain.models.deep_learning.model_config import ModelConfig
from two_towers.infrastructure.config.settings import MLModelSettings
from two_towers.infrastructure.deep_learning.movie_lens_data_manager import DataManager
from two_towers.infrastructure.logging.logger import Logger, setup_logging

logger = Logger.get_logger(__name__)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="PyTorch Two-Tower Recommendation Model")
    parser.add_argument("--config-override", action="store_true", help="Override config with CLI args")
    args = parser.parse_args()
    ml_settings = MLModelSettings()
    config = ModelConfig.from_settings(ml_settings)

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    config.device = str(device)

    logger.info(f"Using configuration: {config}")

    data_manager = DataManager(
        sample_users=config.sample_users,
        device=device,
        sentence_model=config.sentence_model,
    )

    all_ratings, additional_feature_info = data_manager.prepare_data()
    trainer = ModelTrainer(config, all_ratings, additional_feature_info)
    trainer.start_train_pipeline()

    logger.info("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
