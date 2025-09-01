import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional

from feature_processor import FeatureProcessor
from ncf import NCF


class NCFTrainer:
    def __init__(
        self,
        model: NCF,
        train_ratings: pd.DataFrame,
        validation_ratings: pd.DataFrame,
        validation_candidates: Dict,
        feature_processor: FeatureProcessor,
        device: torch.device,
        patience: int = 10,
        min_delta: float = 0.001,
        early_stop_metric: str = "mean_rank",
        scheduler_type: str = "ReduceLROnPlateau",
        scheduler_params: Dict = None,
    ):
        self.model = model
        self.train_ratings = train_ratings
        self.validation_ratings = validation_ratings
        self.validation_candidates = validation_candidates
        self.feature_processor = feature_processor
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        self.early_stop_metric = early_stop_metric

        self.optimizer = torch.optim.Adam(model.parameters())

        if scheduler_params is None:
            scheduler_params = {"mode": "min", "factor": 0.5, "patience": 5}

        if scheduler_type == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, **scheduler_params
            )
        elif scheduler_type == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, **scheduler_params
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=5
            )

        self.best_metric = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {"hit_ratio": [], "mrr": [], "mean_rank": []}

    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            user_input, item_input, labels = [x.to(self.device) for x in batch]
            batch_device = (user_input, item_input, labels)

            self.optimizer.zero_grad()
            loss = self.model.compute_loss(batch_device)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def compute_validation_loss(self, test_ratings, total_users_to_test=None):
        """
        Compute validation loss for a subset of validation data

        Args:
            test_ratings: DataFrame with validation ratings
            precomputed_candidates: Dict of {user_id: [candidate_items]}
            total_users_to_test: Number of users to test
        """
        self.model.eval()
        total_loss = 0
        num_cases = 0

        test_user_item_set = list(
            set(zip(test_ratings["user_id"], test_ratings["movie_id"]))
        )

        try:
            for u, i in tqdm(
                test_user_item_set[:total_users_to_test],
                desc="Computing val loss",
                leave=False,
            ):
                if u not in self.model.feature_processor.user_features_cache:
                    continue

                if i not in self.model.feature_processor.movie_features_cache:
                    continue

                user_feat = (
                    self.model.feature_processor.get_user_features(u)
                    .unsqueeze(0)
                    .to(self.device)
                )
                movie_feat = (
                    self.model.feature_processor.get_movie_features(i)
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    loss = self.model.compute_loss(
                        (user_feat, movie_feat, torch.ones(1).to(self.device))
                    )
                    total_loss += loss.item()
                    num_cases += 1

            if num_cases == 0:
                return 0.0

            return total_loss / num_cases

        except Exception as e:
            print(f"Error computing validation loss: {e}")
            return 0.0

    def validate_model_logic(
        self, test_ratings, precomputed_candidates, total_users_to_test=None, k=10
    ):
        """
        Validation function that works with features instead of IDs

        Args:
            test_ratings: DataFrame with validation ratings
            precomputed_candidates: Dict of {user_id: [candidate_items]}
            total_users_to_test: Number of users to test
            k: Top-k for hit ratio calculation
        """
        test_user_item_set = list(
            set(zip(test_ratings["user_id"], test_ratings["movie_id"]))
        )
        hits = []
        ranks = []
        skipped_cases = 0
        total_cases = 0

        try:
            # Randomly sample users for validation
            if total_users_to_test and len(test_user_item_set) > total_users_to_test:
                sampled_indices = np.random.choice(
                    len(test_user_item_set), size=total_users_to_test, replace=False
                )
                sampled_user_item_set = [test_user_item_set[i] for i in sampled_indices]
            else:
                sampled_user_item_set = test_user_item_set

            for u, i in tqdm(sampled_user_item_set, desc="Validating", leave=False):
                total_cases += 1

                if u not in self.model.feature_processor.user_features_cache:
                    print(f"Skipping user {u} - not in feature cache")
                    skipped_cases += 1
                    continue

                if i not in self.model.feature_processor.movie_features_cache:
                    print(f"Skipping movie {i} - not in feature cache")
                    skipped_cases += 1
                    continue

                candidate_items = precomputed_candidates.get(u, [])

                if i not in candidate_items:
                    candidate_items = candidate_items + [i]

                valid_candidates = [
                    movie_id
                    for movie_id in candidate_items
                    if movie_id in self.model.feature_processor.movie_features_cache
                ]

                if len(valid_candidates) == 0:
                    print(f"No valid candidates for user {u}")
                    skipped_cases += 1
                    continue

                if i not in valid_candidates:
                    print(f"Target movie {i} not in valid candidates for user {u}")
                    skipped_cases += 1
                    continue

                user_feat = (
                    self.model.feature_processor.get_user_features(u)
                    .unsqueeze(0)
                    .to(self.device)
                )

                predicted_scores = []
                for movie_id in valid_candidates:
                    movie_feat = (
                        self.model.feature_processor.get_movie_features(movie_id)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    with torch.no_grad():
                        score = self.model(user_feat, movie_feat).item()
                    predicted_scores.append(score)

                sorted_indices = sorted(
                    range(len(predicted_scores)),
                    key=lambda idx: predicted_scores[idx],
                    reverse=True,
                )
                sorted_items = [valid_candidates[idx] for idx in sorted_indices]
                relevant_item_rank = sorted_items.index(i) + 1
                ranks.append(relevant_item_rank)

                if relevant_item_rank <= k:
                    hits.append(1)
                else:
                    hits.append(0)

            if len(hits) == 0:
                print("Warning: No valid test cases found!")
                return 0.0, 0.0, 0.0

            hit_ratio = sum(hits) / len(hits)
            mrr = sum(1.0 / rank for rank in ranks) / len(ranks)
            mean_rank = sum(ranks) / len(ranks)

            return hit_ratio, mrr, mean_rank

        except Exception as e:
            print(f"Error during validation: {e}")
            return 0.0, 0.0, 0.0

    def validate_model(
        self, test_ratings, precomputed_candidates, total_users_to_test=None, k=10
    ):
        self.model.eval()
        with torch.no_grad():
            hit_ratio, mrr, mean_rank = self.validate_model_logic(
                test_ratings, precomputed_candidates, total_users_to_test, k
            )
        return hit_ratio, mrr, mean_rank

    def check_early_stopping(self, current_metric: float) -> bool:
        if self.early_stop_metric == "mean_rank":
            is_better = current_metric < self.best_metric - self.min_delta
        else:
            is_better = current_metric > self.best_metric + self.min_delta

        if is_better:
            self.best_metric = current_metric
            self.best_epoch = len(self.train_losses)
            self.counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False

    def train(
        self,
        num_epochs: int,
        batch_size: int = 512,
        num_workers: int = 4,
        k: int = 10,
        total_users_to_test: int = None,
        verbose: bool = True,
    ) -> Dict:
        dataloader = self.model.get_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )

        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss = self.train_epoch(dataloader)

            val_loss = self.compute_validation_loss(
                self.validation_ratings, total_users_to_test=total_users_to_test
            )

            hit_ratio, mrr, mean_rank = self.validate_model(
                self.validation_ratings,
                self.validation_candidates,
                total_users_to_test=total_users_to_test,
                k=k,
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics["hit_ratio"].append(hit_ratio)
            self.val_metrics["mrr"].append(mrr)
            self.val_metrics["mean_rank"].append(mean_rank)

            self.scheduler.step(train_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Training Loss: {train_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Hit Ratio @ {k}: {hit_ratio:.3f}")
                print(f"  MRR: {mrr:.3f}")
                print(f"  Mean Rank: {mean_rank:.1f}")
                print("-" * 50)

            current_metric = self.val_metrics[self.early_stop_metric][-1]
            if self.check_early_stopping(current_metric):
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"Best model was at epoch {self.best_epoch + 1}")
                break

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
        }

    def plot_metrics(self, save_path: Optional[str] = None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        axes[0, 0].plot(
            epochs, self.train_losses, "b-", label="Training Loss", linewidth=2
        )
        axes[0, 0].plot(
            epochs, self.val_losses, "r--", label="Validation Loss", linewidth=2
        )
        axes[0, 0].set_title("Training vs Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(
            epochs,
            self.val_metrics["hit_ratio"],
            "g-",
            label="Hit Ratio @ 10",
            linewidth=2,
        )
        axes[0, 1].set_title("Hit Ratio @ 10")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Hit Ratio")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[1, 0].plot(epochs, self.val_metrics["mrr"], "r-", label="MRR", linewidth=2)
        axes[1, 0].set_title("Mean Reciprocal Rank")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("MRR")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        axes[1, 1].plot(
            epochs, self.val_metrics["mean_rank"], "m-", label="Mean Rank", linewidth=2
        )
        axes[1, 1].set_title("Mean Rank")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Mean Rank")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Metrics plot saved to {save_path}")

        plt.show()

    def restore_best_model(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model from epoch {self.best_epoch + 1}")
        else:
            print("No best model state available")

    def save_model(self, filepath: str, save_best: bool = True):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if save_best and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Saving best model from epoch {self.best_epoch + 1}")

        self.model.save_weights(filepath)
        print(f"Model saved to {filepath}")

    def get_best_metrics(self) -> Dict:
        return {
            "best_epoch": self.best_epoch + 1,
            "best_mean_rank": self.best_metric,
            "best_hit_ratio": self.val_metrics["hit_ratio"][self.best_epoch],
            "best_mrr": self.val_metrics["mrr"][self.best_epoch],
        }
