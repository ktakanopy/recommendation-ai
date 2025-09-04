import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MovieLensDataset(Dataset):
    def __init__(
        self,
        ratings,
        precomputed_candidates,
        feature_processor,
        num_negatives=4,
        return_ids=False,
        all_negatives=False,
    ):
        self.ratings = ratings
        self.precomputed_candidates = precomputed_candidates
        self.feature_processor = feature_processor
        self.num_negatives = num_negatives
        self.return_ids = return_ids
        self.all_negatives = all_negatives
        self.user_ids, self.movie_ids, self.user_features, self.movie_features, self.labels = self.get_dataset()

    def __len__(self):
        return len(self.user_features)

    def __getitem__(self, idx):
        if self.return_ids:
            return self.user_ids[idx], self.movie_ids[idx], self.user_features[idx], self.movie_features[idx], self.labels[idx]
        return self.user_features[idx], self.movie_features[idx], self.labels[idx]

    def get_dataset(self):
        user_id_list, movie_id_list, user_features_list, movie_features_list, labels = [], [], [], [], []

        user_positive_items = (
            self.ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
        )

        for user_id, positive_items in tqdm(
            user_positive_items.items(), desc="Processing movie dataset"
        ):
            if user_id not in self.precomputed_candidates:
                continue

            user_feat = self.feature_processor.get_user_features(user_id)
            available_candidates = self.precomputed_candidates[user_id]

            # Add positive samples
            for item_id in positive_items:
                movie_feat = self.feature_processor.get_movie_features(item_id)
                user_id_list.append(user_id)
                movie_id_list.append(item_id)
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(1)

            # Add negative samples using precomputed candidates
            self._add_negative_samples(
                user_id_list,
                movie_id_list,
                user_features_list,
                movie_features_list,
                labels,
                user_id,
                user_feat,
                positive_items,
                available_candidates,
            )

        user_features_tensor = torch.stack(user_features_list)
        movie_features_tensor = torch.stack(movie_features_list)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return user_id_list, movie_id_list, user_features_tensor, movie_features_tensor, labels_tensor

    def _add_negative_samples(
        self,
        user_id_list,
        movie_id_list,
        user_features_list,
        movie_features_list,
        labels,
        user_id,
        user_feat,
        positive_items,
        available_candidates,
    ):
        """Add negative samples using precomputed candidates"""
        negative_candidates = [item for item in available_candidates if item not in positive_items]

        if self.all_negatives:
            for neg_item in negative_candidates:
                movie_feat = self.feature_processor.get_movie_features(neg_item)
                user_id_list.append(user_id)
                movie_id_list.append(int(neg_item))
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(0)
        else:
            total_negatives_needed = len(positive_items) * self.num_negatives
            if len(negative_candidates) > 0:
                sample_size = min(total_negatives_needed, len(negative_candidates))
                sampled_negatives = np.random.choice(
                    negative_candidates, size=sample_size, replace=False
                )
                for neg_item in sampled_negatives:
                    movie_feat = self.feature_processor.get_movie_features(neg_item)
                    user_id_list.append(user_id)
                    movie_id_list.append(int(neg_item))
                    user_features_list.append(user_feat)
                    movie_features_list.append(movie_feat)
                    labels.append(0)
