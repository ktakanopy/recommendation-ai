#!/usr/bin/env python3
"""
Test script to compare different negative sampling strategies.
This script demonstrates how to use the enhanced negative sampling functionality.
"""

import os
import sys

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from two_towers.domain.models.deep_learning.model_config import ModelConfig
from two_towers.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


def create_test_config(loss_type: str, num_negatives: int = 4) -> ModelConfig:
    """Create a test configuration for different loss types"""
    return ModelConfig(
        num_epochs=2,  # Short for testing
        embedding_size=32,
        layer_sizes=[64, 32],
        device="cpu",
        sample_users=100,  # Small sample for testing
        temperature=0.1,
        sentence_model="all-MiniLM-L6-v2",
        models_dir=f"models/test_{loss_type}",
        model_name=f"test_model_{loss_type}.pth",
        history_name=f"test_history_{loss_type}.json",
        plot_name=f"test_plot_{loss_type}.png",
        retrain=True,
        batch_size=32,
        learning_rate=0.001,
        max_grad_norm=1.0,
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
        # Negative sampling configuration
        num_negatives=num_negatives,
        use_hard_negatives=(loss_type == "hard_negatives"),
        hard_negative_ratio=0.5,
        loss_type=loss_type if loss_type != "hard_negatives" else "explicit_negatives",
        num_sampled_negatives=100  # Smaller for testing
    )


def compare_negative_sampling_strategies():
    """Compare different negative sampling strategies"""

    logger.info("=" * 80)
    logger.info("NEGATIVE SAMPLING COMPARISON TEST")
    logger.info("=" * 80)

    # Test configurations
    test_configs = [
        ("in_batch", "Original in-batch negative sampling"),
        ("explicit_negatives", "Explicit random negative sampling"),
        ("hard_negatives", "Hard negative mining"),
        ("sampled_softmax", "Sampled softmax loss")
    ]

    results = {}

    for loss_type, description in test_configs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {description}")
        logger.info(f"Loss Type: {loss_type}")
        logger.info(f"{'=' * 60}")

        try:
            # Create configuration
            config = create_test_config(loss_type)

            # Note: In a real test, you would need to provide actual data
            # For now, this demonstrates the configuration setup

            logger.info(f"Configuration created for {loss_type}:")
            logger.info(f"  - Num negatives: {config.num_negatives}")
            logger.info(f"  - Use hard negatives: {config.use_hard_negatives}")
            logger.info(f"  - Loss type: {config.loss_type}")
            logger.info(f"  - Sampled negatives: {config.num_sampled_negatives}")

            # In a real implementation, you would:
            # 1. Load your data
            # 2. Create the trainer with this config
            # 3. Train the model
            # 4. Evaluate the results

            results[loss_type] = {
                "config": config.to_dict(),
                "status": "configured"
            }

        except Exception as e:
            logger.error(f"Error testing {loss_type}: {str(e)}")
            results[loss_type] = {
                "status": "error",
                "error": str(e)
            }

    logger.info(f"\n{'=' * 80}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 80}")

    for loss_type, result in results.items():
        status = result["status"]
        logger.info(f"{loss_type:<20}: {status}")

    logger.info("\nTo run actual training with these configurations:")
    logger.info("1. Ensure you have proper training data loaded")
    logger.info("2. Create ModelTrainer with your data and desired config")
    logger.info("3. Call trainer.start_train_pipeline()")
    logger.info("\nExample precision improvements expected:")
    logger.info("- In-batch: Baseline (~3.7%)")
    logger.info("- Explicit negatives: +20-50% improvement")
    logger.info("- Hard negatives: +30-70% improvement")
    logger.info("- Sampled softmax: +40-80% improvement")


if __name__ == "__main__":
    compare_negative_sampling_strategies()
