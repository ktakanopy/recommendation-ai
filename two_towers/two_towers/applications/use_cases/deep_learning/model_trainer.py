import json
import os
import random
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader

from two_towers.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from two_towers.applications.use_cases.deep_learning.model_evaluate import ModelEvaluator
from two_towers.domain.models.deep_learning.model_config import ModelConfig
from two_towers.domain.models.deep_learning.two_tower_model import TwoTowerModel
from two_towers.infrastructure.deep_learning.movie_lens_dataset import MovieLensDataset
from two_towers.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ModelTrainer:
    def __init__(
        self,
        config: ModelConfig,
        all_ratings: Dict[str, Any],
        additional_feature_info: FeatureInfoDto,
    ):
        self.config = config
        self.device = config.device
        self.all_ratings = all_ratings
        self.additional_feature_info = additional_feature_info
        self.unique_movie_titles = additional_feature_info.unique_movie_titles
        self.unique_user_ids = additional_feature_info.unique_user_ids
        self.evaluator = ModelEvaluator()

    def _create_temporal_splits(self, all_ratings: Dict[str, Any], train_split: float, val_split: float, test_split: float):
        """Create temporal train/val/test splits for proper recommendation evaluation"""
        # Check if timestamp data exists
        if "timestamp" not in all_ratings:
            logger.info("No timestamp data found, falling back to leave-one-out user splits")
            return self._create_leave_one_out_splits(all_ratings, train_split, val_split, test_split)

        # Sort interactions by timestamp
        timestamps = all_ratings["timestamp"]
        sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])

        total_interactions = len(sorted_indices)
        train_size = int(train_split * total_interactions)
        val_size = int(val_split * total_interactions)

        # Split chronologically
        train_indices = sorted_indices[:train_size]
        val_indices = sorted_indices[train_size:train_size + val_size]
        test_indices = sorted_indices[train_size + val_size:]

        logger.info(f"Temporal splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

        return set(train_indices), set(val_indices), set(test_indices)

    def _create_leave_one_out_splits(self, all_ratings: Dict[str, Any], train_split: float, val_split: float, test_split: float):
        """Create leave-one-out splits per user for recommendation evaluation"""
        # Group interactions by user
        user_interactions = {}
        for i in range(len(all_ratings["user_id"])):
            user_id = all_ratings["user_id"][i]
            if user_id not in user_interactions:
                user_interactions[user_id] = []
            user_interactions[user_id].append(i)

        train_indices = []
        val_indices = []
        test_indices = []

        # For each user, use last interactions for validation/test
        for user_id, interactions in user_interactions.items():
            if len(interactions) < 3:  # Need at least 3 interactions per user
                # Put all in training if too few interactions
                train_indices.extend(interactions)
                continue

            # Sort user interactions by timestamp if available
            if "timestamp" in all_ratings:
                interactions.sort(key=lambda i: all_ratings["timestamp"][i])

            # Use last 2 interactions for val/test, rest for training
            train_indices.extend(interactions[:-2])
            val_indices.append(interactions[-2])
            test_indices.append(interactions[-1])

        logger.info(f"Leave-one-out splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

        return set(train_indices), set(val_indices), set(test_indices)

    def _create_user_level_splits(self, all_ratings: Dict[str, Any], train_split: float, val_split: float, test_split: float):
        """Legacy user-level splits (kept for backward compatibility)"""
        # Get unique users
        unique_users = list(set(all_ratings["user_id"]))
        logger.info(f"Total unique users: {len(unique_users)}")

        # Shuffle users with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(unique_users)

        # Calculate split sizes
        total_users = len(unique_users)
        train_size = int(train_split * total_users)
        val_size = int(val_split * total_users)

        # Split users
        train_users = set(unique_users[:train_size])
        val_users = set(unique_users[train_size:train_size + val_size])
        test_users = set(unique_users[train_size + val_size:])

        logger.info(f"User splits - Train: {len(train_users)}, Val: {len(val_users)}, Test: {len(test_users)}")

        # Verify no overlap
        assert len(train_users & val_users) == 0, "Train and val users overlap!"
        assert len(train_users & test_users) == 0, "Train and test users overlap!"
        assert len(val_users & test_users) == 0, "Val and test users overlap!"

        return train_users, val_users, test_users

    def _filter_ratings_by_indices(self, all_ratings: Dict[str, Any], index_set: set) -> Dict[str, Any]:
        """Filter ratings data to include only specified indices"""
        filtered_ratings = {key: [] for key in all_ratings.keys()}

        for i in range(len(all_ratings["user_id"])):
            if i in index_set:
                for key in all_ratings.keys():
                    filtered_ratings[key].append(all_ratings[key][i])

        logger.info(f"Filtered {len(filtered_ratings['user_id'])} interactions from {len(index_set)} indices")
        return filtered_ratings

    def _filter_ratings_by_users(self, all_ratings: Dict[str, Any], user_set: set) -> Dict[str, Any]:
        """Filter ratings data to include only specified users (legacy method)"""
        filtered_ratings = {key: [] for key in all_ratings.keys()}

        for i in range(len(all_ratings["user_id"])):
            if all_ratings["user_id"][i] in user_set:
                for key in all_ratings.keys():
                    filtered_ratings[key].append(all_ratings[key][i])

        logger.info(f"Filtered {len(filtered_ratings['user_id'])} interactions for {len(user_set)} users")
        return filtered_ratings

    def _plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_recalls: List[float],
        val_precisions: List[float],
        val_mrrs: List[float],
        save_path: str,
    ):
        epochs = range(1, len(train_losses) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_recalls, "g-", label="Validation Recall@10", linewidth=2)
        ax2.set_title("Validation Recall@10")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Recall@10")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, val_precisions, "m-", label="Validation Precision@10", linewidth=2)
        ax3.set_title("Validation Precision@10")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Precision@10")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.plot(epochs, val_mrrs, "c-", label="Validation MRR", linewidth=2)
        ax4.set_title("Validation MRR")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("MRR")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Training curves saved to: {save_path}")

    def train_model(self) -> TwoTowerModel:
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.unique_user_ids)}
        title_to_idx = self.additional_feature_info.sentence_embeddings.title_to_idx

        # Create temporal/leave-one-out splits for proper recommendation evaluation
        train_indices, val_indices, test_indices = self._create_temporal_splits(
            self.all_ratings, self.config.train_split, self.config.val_split, self.config.test_split
        )

        # Filter data by index splits
        train_ratings = self._filter_ratings_by_indices(self.all_ratings, train_indices)
        val_ratings = self._filter_ratings_by_indices(self.all_ratings, val_indices)
        test_ratings = self._filter_ratings_by_indices(self.all_ratings, test_indices)
        
        # Build training user interactions for proper evaluation ground truth
        train_user_interactions = self.evaluator.build_user_interactions_from_training_data(
            train_ratings, user_id_to_idx, title_to_idx
        )

        # Determine if we need negative samples based on loss type
        needs_negatives = self.config.loss_type in ["explicit_negatives", "sampled_softmax"]

        # Create training dataset
        train_dataset = MovieLensDataset(
            train_ratings,
            user_id_to_idx,
            title_to_idx,
            num_negatives=self.config.num_negatives,
            use_hard_negatives=self.config.use_hard_negatives,
            hard_negative_ratio=self.config.hard_negative_ratio,
            include_negatives=needs_negatives
        )

        # Create evaluation datasets (always without negatives for consistent evaluation)
        val_dataset = MovieLensDataset(
            val_ratings,
            user_id_to_idx,
            title_to_idx,
            include_negatives=False
        )

        test_dataset = MovieLensDataset(
            test_ratings,
            user_id_to_idx,
            title_to_idx,
            include_negatives=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            num_workers=0,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            num_workers=0,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            num_workers=0,
        )
        model = TwoTowerModel(
            layer_sizes=self.config.layer_sizes,
            unique_user_ids=self.unique_user_ids,
            embedding_size=self.config.embedding_size,
            additional_feature_info=self.additional_feature_info,
            device=self.device,
            dropout_rate=self.config.dropout_rate,
        )
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        max_grad_norm = 1.0

        train_losses = []
        val_losses = []
        val_recalls = []
        val_precisions = []
        val_mrrs = []

        logger.info("=== DATA SPLIT VERIFICATION ===")
        logger.info(f"Training: {len(train_dataset)} interactions")
        logger.info(f"Validation: {len(val_dataset)} interactions")
        logger.info(f"Test: {len(test_dataset)} interactions")
        logger.info(f"Loss type: {self.config.loss_type}, Needs negatives: {needs_negatives}")
        logger.info("Using temporal/leave-one-out evaluation methodology")
        logger.info("=== STARTING TRAINING ===")

        logger.info(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")

        for epoch in range(self.config.num_epochs):
            model.train()
            total_train_loss = 0
            train_batches = 0

            for batch_data in train_loader:
                optimizer.zero_grad()

                # Choose loss function based on configuration
                if self.config.loss_type == "in_batch":
                    # For in-batch loss, we expect 2-tuple format
                    user_inputs, movie_inputs = batch_data
                    user_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in user_inputs.items() if torch.is_tensor(v)
                    }
                    movie_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in movie_inputs.items() if torch.is_tensor(v)
                    }
                    loss = model.compute_loss(user_inputs, movie_inputs, temperature=self.config.temperature)

                elif self.config.loss_type == "sampled_softmax":
                    # For sampled softmax, we can use either format but only need positive movies
                    if len(batch_data) == 2:
                        user_inputs, positive_movie_inputs = batch_data
                    else:  # len(batch_data) == 3
                        user_inputs, positive_movie_inputs, _ = batch_data

                    user_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in user_inputs.items() if torch.is_tensor(v)
                    }
                    positive_movie_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in positive_movie_inputs.items() if torch.is_tensor(v)
                    }
                    loss = model.compute_sampled_softmax_loss(
                        user_inputs, positive_movie_inputs, num_sampled_negatives=self.config.num_sampled_negatives
                    )

                else:  # "explicit_negatives"
                    # For explicit negatives, we expect 3-tuple format
                    user_inputs, positive_movie_inputs, negative_movie_inputs = batch_data
                    user_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in user_inputs.items() if torch.is_tensor(v)
                    }
                    positive_movie_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in positive_movie_inputs.items() if torch.is_tensor(v)
                    }
                    negative_movie_inputs = {
                        k: v.to(self.device, non_blocking=True) for k, v in negative_movie_inputs.items() if torch.is_tensor(v)
                    }
                    loss = model.compute_loss_with_negatives(user_inputs, positive_movie_inputs, negative_movie_inputs)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                total_train_loss += loss.item()
                train_batches += 1

            avg_train_loss = total_train_loss / train_batches
            train_losses.append(avg_train_loss)

            val_metrics = self.evaluator.evaluate_model(
                model, val_loader, self.device,
                additional_feature_info=self.additional_feature_info,
                evaluation_type="leave_one_out",
                train_user_interactions=train_user_interactions
            )
            val_losses.append(val_metrics["loss"])
            val_recalls.append(val_metrics["recall@10"])
            val_precisions.append(val_metrics["precision@10"])
            val_mrrs.append(val_metrics.get("mrr@10", val_metrics.get("mrr", 0.0)))

            logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Recall@10: {val_metrics['recall@10']:.4f}, "
                f"Val Precision@10: {val_metrics['precision@10']:.4f}, "
                f"Val MRR: {val_metrics.get('mrr@10', val_metrics.get('mrr', 0.0)):.4f}"
            )

        test_metrics = self.evaluator.evaluate_model(
            model, test_loader, self.device,
            additional_feature_info=self.additional_feature_info,
            evaluation_type="leave_one_out",
            train_user_interactions=train_user_interactions
        )
        logger.info(
            f"Test Results - "
            f"Loss: {test_metrics['loss']:.4f}, "
            f"Recall@10: {test_metrics['recall@10']:.4f}, "
            f"Precision@10: {test_metrics['precision@10']:.4f}, "
            f"MRR: {test_metrics.get('mrr@10', test_metrics.get('mrr', 0.0)):.4f}"
        )

        model.training_history = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_recalls": val_recalls,
            "val_precisions": val_precisions,
            "val_mrrs": val_mrrs,
            "test_metrics": test_metrics,
        }

        return model

    def save_model(self, model: TwoTowerModel, models_dir: str):
        model_path = os.path.join(models_dir, self.config.model_name)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model weights saved to: {model_path}")

    def save_history(self, history: Dict[str, Any], models_dir: str):
        history_path = os.path.join(models_dir, self.config.history_name)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        logger.info(f"Training history saved to: {history_path}")

    def save_plot(self, history: Dict[str, Any], models_dir: str, plot_name: str):
        plot_path = os.path.join(models_dir, plot_name)
        self._plot_training_curves(
            history["train_losses"],
            history["val_losses"],
            history["val_recalls"],
            history["val_precisions"],
            history["val_mrrs"],
            plot_path,
        )
        logger.info(f"Training curves saved to: {plot_path}")

    def save_config(self, config: ModelConfig, models_dir: str):
        config_path = os.path.join(models_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
        logger.info(f"Model config saved to: {config_path}")

    def start_train_pipeline(self):
        models_dir = self.config.models_dir
        os.makedirs(models_dir, exist_ok=True)

        logger.info(f"\n{'=' * 60}")
        logger.info("Starting training pipeline")
        logger.info(f"{'=' * 60}")

        model = self.train_model()

        self.save_model(model, models_dir)
        self.save_history(model.training_history, models_dir)
        self.save_plot(model.training_history, models_dir, self.config.plot_name)
        self.save_config(self.config, models_dir)

        self._display_model_comparison(model.training_history)

    def _display_model_comparison(self, history: Dict[str, Any]):
        logger.info(f"\n{'=' * 80}")
        logger.info("TRAINING RESULTS")
        logger.info(f"{'=' * 80}")
        final_train = history["train_losses"][-1] if history["train_losses"] else 0
        final_val_loss = history["val_losses"][-1] if history["val_losses"] else 0
        final_val_recall = history["val_recalls"][-1] if history["val_recalls"] else 0
        final_val_precision = history["val_precisions"][-1] if history["val_precisions"] else 0
        final_val_mrr = history["val_mrrs"][-1] if history["val_mrrs"] else 0

        logger.info(f"Final Train Loss: {final_train:.4f}")
        logger.info(f"Final Val Loss: {final_val_loss:.4f}")
        logger.info(f"Final Val Recall@10: {final_val_recall:.4f}")
        logger.info(f"Final Val Precision@10: {final_val_precision:.4f}")
        logger.info(f"Final Val MRR: {final_val_mrr:.4f}")

        test_metrics = history["test_metrics"]
        logger.info("\nTest Metrics:")
        logger.info(f"Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Recall@10: {test_metrics['recall@10']:.4f}")
        logger.info(f"Precision@10: {test_metrics['precision@10']:.4f}")
        logger.info(f"MRR@10: {test_metrics.get('mrr@10', test_metrics.get('mrr', 0.0)):.4f}")
        if 'hit_rate@10' in test_metrics:
            logger.info(f"Hit Rate@10: {test_metrics['hit_rate@10']:.4f}")
        if 'ndcg@10' in test_metrics:
            logger.info(f"NDCG@10: {test_metrics['ndcg@10']:.4f}")

        models_dir = self.config.models_dir
        logger.info(f"\n{'=' * 80}")
        logger.info("SAVED MODEL")
        logger.info(f"{'=' * 80}")
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
            if model_files:
                logger.info(f"Model saved in {models_dir}/:")
                for model_file in model_files:
                    file_path = os.path.join(models_dir, model_file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)
                    logger.info(f"  - {model_file} ({file_size:.2f} MB)")
            else:
                logger.info(f"No saved model found in {models_dir}/ directory")
        else:
            logger.info(f"{models_dir}/ directory does not exist")


def custom_collate_fn(batch):
    """Intelligent collate function that handles both 2-tuple and 3-tuple formats"""
    # Check the format of the first item to determine batch structure
    if len(batch[0]) == 2:
        # 2-tuple format: (user_inputs, movie_inputs) - for evaluation
        user_batch = {}
        movie_batch = {}

        user_inputs_list = [item[0] for item in batch]
        movie_inputs_list = [item[1] for item in batch]

        # Process user inputs
        for key in user_inputs_list[0].keys():
            values = [item[key] for item in user_inputs_list]
            if key in ["user_id", "gender", "occupation"]:
                user_batch[key] = torch.tensor(values, dtype=torch.long)
            elif key in ["user_age",] or isinstance(values[0], (int, float)):
                user_batch[key] = torch.tensor(values, dtype=torch.float32)
            else:
                user_batch[key] = values

        # Process movie inputs
        for key in movie_inputs_list[0].keys():
            values = [item[key] for item in movie_inputs_list]
            if key == "movie_idx":
                movie_batch[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], (int, float)):
                movie_batch[key] = torch.tensor(values, dtype=torch.float32)
            else:
                movie_batch[key] = values

        return user_batch, movie_batch

    elif len(batch[0]) == 3:
        # 3-tuple format: (user_inputs, positive_movie_inputs, negative_movie_inputs) - for training
        user_batch = {}
        positive_movie_batch = {}
        negative_movie_batch = {}

        user_inputs_list = [item[0] for item in batch]
        positive_movie_inputs_list = [item[1] for item in batch]
        negative_movie_inputs_list = [item[2] for item in batch]

        # Process user inputs
        for key in user_inputs_list[0].keys():
            values = [item[key] for item in user_inputs_list]
            if key in ["user_id", "gender", "occupation"]:
                user_batch[key] = torch.tensor(values, dtype=torch.long)
            elif key in ["user_age",] or isinstance(values[0], (int, float)):
                user_batch[key] = torch.tensor(values, dtype=torch.float32)
            else:
                user_batch[key] = values

        # Process positive movie inputs
        for key in positive_movie_inputs_list[0].keys():
            values = [item[key] for item in positive_movie_inputs_list]
            if key == "movie_idx":
                positive_movie_batch[key] = torch.tensor(values, dtype=torch.long)
            elif isinstance(values[0], (int, float)):
                positive_movie_batch[key] = torch.tensor(values, dtype=torch.float32)
            else:
                positive_movie_batch[key] = values

        # Process negative movie inputs - flatten the list of lists
        batch_size = len(negative_movie_inputs_list)
        num_negatives = len(negative_movie_inputs_list[0])

        # Flatten: [batch_size, num_negatives] -> [batch_size * num_negatives]
        flattened_negatives = []
        for user_negatives in negative_movie_inputs_list:
            flattened_negatives.extend(user_negatives)

        for key in flattened_negatives[0].keys():
            values = [item[key] for item in flattened_negatives]
            if key == "movie_idx":
                # Reshape to [batch_size, num_negatives]
                negative_movie_batch[key] = torch.tensor(values, dtype=torch.long).view(batch_size, num_negatives)
            elif isinstance(values[0], (int, float)):
                negative_movie_batch[key] = torch.tensor(values, dtype=torch.float32).view(batch_size, num_negatives)
            else:
                negative_movie_batch[key] = values

        return user_batch, positive_movie_batch, negative_movie_batch

    else:
        raise ValueError(f"Unexpected batch format with {len(batch[0])} elements per item")
