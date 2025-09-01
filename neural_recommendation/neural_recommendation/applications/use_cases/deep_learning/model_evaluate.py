from typing import Any, Dict, List, Set

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from neural_recommendation.domain.models.deep_learning.two_tower_model import TwoTowerModel


class ModelEvaluator:
    @staticmethod
    def calculate_recommendation_metrics(
        user_embeddings: torch.Tensor,
        all_movie_embeddings: torch.Tensor,
        ground_truth_movies: List[Set[int]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate proper recommendation metrics by ranking against full catalog.
        
        Args:
            user_embeddings: [batch_size, embed_dim] - User embeddings
            all_movie_embeddings: [num_movies, embed_dim] - All movie embeddings
            ground_truth_movies: List of sets, each containing movie indices user actually liked
            k: Top-k for evaluation
        """
        batch_size = user_embeddings.size(0)

        # Compute similarities between users and ALL movies
        similarities = torch.matmul(user_embeddings, all_movie_embeddings.T)  # [batch_size, num_movies]

        # Get top-k predictions for each user
        _, top_k_indices = torch.topk(similarities, k=k, dim=1)  # [batch_size, k]

        total_recall = 0.0
        total_precision = 0.0
        total_mrr = 0.0
        total_ndcg = 0.0
        total_hit_rate = 0.0

        for i in range(batch_size):
            if not ground_truth_movies[i]:  # Skip users with no ground truth
                continue

            user_top_k = set(top_k_indices[i].cpu().tolist())
            user_relevant = ground_truth_movies[i]

            # Calculate hits (intersection of recommended and relevant)
            hits = user_top_k.intersection(user_relevant)
            num_hits = len(hits)

            # Recall@K: fraction of relevant items that were recommended
            recall = num_hits / len(user_relevant) if user_relevant else 0.0
            total_recall += recall

            # Precision@K: fraction of recommended items that were relevant
            precision = num_hits / k
            total_precision += precision

            # Hit Rate: whether at least one relevant item was recommended
            hit_rate = 1.0 if num_hits > 0 else 0.0
            total_hit_rate += hit_rate

            # MRR: Mean Reciprocal Rank of first relevant item
            mrr = 0.0
            for rank, movie_idx in enumerate(top_k_indices[i].cpu().tolist(), 1):
                if movie_idx in user_relevant:
                    mrr = 1.0 / rank
                    break
            total_mrr += mrr

            # NDCG@K: Normalized Discounted Cumulative Gain
            ndcg = ModelEvaluator._calculate_ndcg(
                top_k_indices[i].cpu().tolist(), user_relevant, k
            )
            total_ndcg += ndcg

        # Average across all users
        num_users = batch_size
        return {
            f"recall@{k}": total_recall / num_users,
            f"precision@{k}": total_precision / num_users,
            f"hit_rate@{k}": total_hit_rate / num_users,
            f"mrr@{k}": total_mrr / num_users,
            f"ndcg@{k}": total_ndcg / num_users,
        }

    @staticmethod
    def _calculate_ndcg(predictions: List[int], relevant_items: Set[int], k: int) -> float:
        """Calculate NDCG@K for a single user"""
        # DCG: sum of (relevance / log2(rank + 1))
        dcg = 0.0
        for i, movie_idx in enumerate(predictions[:k]):
            if movie_idx in relevant_items:
                dcg += 1.0 / torch.log2(torch.tensor(i + 2.0)).item()  # +2 because rank starts at 1

        # IDCG: DCG of perfect ranking (all relevant items at top)
        idcg = 0.0
        for i in range(min(len(relevant_items), k)):
            idcg += 1.0 / torch.log2(torch.tensor(i + 2.0)).item()

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def evaluate_model(
        model: TwoTowerModel,
        test_loader: DataLoader,
        device: str,
        additional_feature_info: Dict = None,
        evaluation_type: str = "leave_one_out",
        train_user_interactions: Dict[int, Set[int]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model using proper recommendation evaluation methodology.
        
        Args:
            model: The two-tower model to evaluate
            test_loader: DataLoader with test data
            device: Device to run evaluation on
            additional_feature_info: Feature info containing movie catalog
            evaluation_type: Type of evaluation ("leave_one_out" or "temporal")
            train_user_interactions: User interactions from TRAINING data for ground truth
        """
        model.eval()
        total_loss = 0
        batch_count = 0

        # Precompute all movie embeddings for ranking
        if additional_feature_info is None:
            # Fallback to old evaluation method if no feature info provided
            return ModelEvaluator._evaluate_model_legacy(model, test_loader, device)

        all_movie_embeddings = ModelEvaluator._get_all_movie_embeddings(
            model, additional_feature_info, device
        )

        # Use training data for ground truth to avoid data leakage
        if train_user_interactions is None:
            # Fallback to old method if training interactions not provided
            # NOTE: This is still problematic as it uses test data for ground truth
            user_interactions = ModelEvaluator._build_user_interactions_from_loader(test_loader)
        else:
            user_interactions = train_user_interactions

        total_metrics = {
            "recall@10": 0.0, "precision@10": 0.0, "hit_rate@10": 0.0,
            "mrr@10": 0.0, "ndcg@10": 0.0
        }

        with torch.no_grad():
            for user_inputs, movie_inputs in test_loader:
                user_inputs = {k: v.to(device, non_blocking=True) for k, v in user_inputs.items() if torch.is_tensor(v)}
                movie_inputs = {k: v.to(device, non_blocking=True) for k, v in movie_inputs.items() if torch.is_tensor(v)}

                # Calculate loss (keeping original loss computation)
                loss = model.compute_loss(user_inputs, movie_inputs)
                total_loss += loss.item()
                batch_count += 1

                # Get user embeddings
                user_embeddings = model.user_model(user_inputs)
                user_embeddings = model.query_tower(user_embeddings)
                user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

                # Prepare ground truth for this batch
                batch_ground_truth = []
                for i in range(len(user_inputs["user_id"])):
                    user_id = user_inputs["user_id"][i].item()
                    
                    if evaluation_type == "leave_one_out":
                        # For leave-one-out: ground truth is the test movie user is interacting with
                        # But we need to filter this against training interactions to avoid cold-start
                        test_movie = movie_inputs["movie_idx"][i].item()
                        
                        # Check if user has training history
                        if user_id in user_interactions and len(user_interactions[user_id]) > 0:
                            # User has training history, use test movie as ground truth
                            batch_ground_truth.append({test_movie})
                        else:
                            # Cold-start user, no ground truth available
                            batch_ground_truth.append(set())
                    else:
                        # For temporal evaluation, use training interactions as ground truth
                        batch_ground_truth.append(user_interactions.get(user_id, set()))

                # Calculate recommendation metrics
                batch_metrics = ModelEvaluator.calculate_recommendation_metrics(
                    user_embeddings, all_movie_embeddings, batch_ground_truth, k=10
                )

                for metric, value in batch_metrics.items():
                    total_metrics[metric] += value

        avg_loss = total_loss / batch_count if batch_count > 0 else float("inf")
        avg_metrics = {metric: value / batch_count for metric, value in total_metrics.items()}
        avg_metrics["loss"] = avg_loss

        return avg_metrics

    @staticmethod
    def _get_all_movie_embeddings(model, additional_feature_info, device) -> torch.Tensor:
        """Precompute embeddings for all movies in the catalog"""
        title_to_idx = additional_feature_info.sentence_embeddings.title_to_idx
        num_movies = len(title_to_idx)

        all_movie_indices = torch.arange(num_movies, device=device)
        movie_inputs = {"movie_idx": all_movie_indices}

        with torch.no_grad():
            movie_embeddings = model.movie_model(movie_inputs)
            movie_embeddings = model.candidate_tower(movie_embeddings)
            movie_embeddings = F.normalize(movie_embeddings, p=2, dim=1)

        return movie_embeddings

    @staticmethod
    def _build_user_interactions_from_loader(data_loader) -> Dict[int, Set[int]]:
        """Build user interaction history from data loader"""
        user_interactions = {}
        for user_inputs, movie_inputs in data_loader:
            for i in range(len(user_inputs["user_id"])):
                user_id = user_inputs["user_id"][i].item()
                movie_idx = movie_inputs["movie_idx"][i].item()

                if user_id not in user_interactions:
                    user_interactions[user_id] = set()
                user_interactions[user_id].add(movie_idx)

        return user_interactions
    
    @staticmethod
    def build_user_interactions_from_training_data(train_ratings: Dict[str, Any], user_id_to_idx: Dict[str, int], title_to_idx: Dict[str, int]) -> Dict[int, Set[int]]:
        """Build user interaction history from raw training data to avoid data leakage"""
        user_interactions = {}
        
        for i in range(len(train_ratings["user_id"])):
            user_id_str = train_ratings["user_id"][i]
            movie_title = train_ratings["movie_title"][i]
            
            # Convert to indices
            user_id = user_id_to_idx.get(user_id_str, -1)
            movie_idx = title_to_idx.get(movie_title, -1)
            
            if user_id != -1 and movie_idx != -1:
                if user_id not in user_interactions:
                    user_interactions[user_id] = set()
                user_interactions[user_id].add(movie_idx)
        
        return user_interactions

    @staticmethod
    def _evaluate_model_legacy(model: TwoTowerModel, test_loader: DataLoader, device: str) -> Dict[str, float]:
        """Legacy evaluation method for backward compatibility"""
        model.eval()
        total_loss = 0
        batch_count = 0
        total_metrics = {"recall@10": 0.0, "precision@10": 0.0, "mrr@10": 0.0}

        with torch.no_grad():
            for user_inputs, movie_inputs in test_loader:
                user_inputs = {k: v.to(device, non_blocking=True) for k, v in user_inputs.items() if torch.is_tensor(v)}
                movie_inputs = {k: v.to(device, non_blocking=True) for k, v in movie_inputs.items() if torch.is_tensor(v)}

                loss = model.compute_loss(user_inputs, movie_inputs)
                total_loss += loss.item()
                batch_count += 1

                query_embeddings, candidate_embeddings = model(user_inputs, movie_inputs)
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)

                # Old method: in-batch evaluation (kept for compatibility)
                similarities = torch.matmul(query_embeddings, candidate_embeddings.T)
                batch_size = similarities.size(0)

                # Simple diagonal evaluation (legacy)
                for i in range(batch_size):
                    _, top_k = torch.topk(similarities[i], k=10)
                    if i in top_k:
                        total_metrics["recall@10"] += 1.0 / batch_size
                        total_metrics["precision@10"] += 1.0 / (10 * batch_size)

                    rank = (similarities[i].argsort(descending=True) == i).nonzero().item() + 1
                    total_metrics["mrr@10"] += (1.0 / rank) / batch_size

        avg_loss = total_loss / batch_count if batch_count > 0 else float("inf")
        avg_metrics = {metric: value / batch_count for metric, value in total_metrics.items()}
        avg_metrics["loss"] = avg_loss

        return avg_metrics
