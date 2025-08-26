

from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

class MovieLensTrainDataset(Dataset):
    """Optimized MovieLens Dataset using Precomputed Candidates for Negative Sampling with Features
    Args:
        ratings(pd.DataFrame): Dataframe containing the movie ratings
        precomputed_candidates(dict): Precomputed candidates for each user {user_id: [candidate_items]}
        feature_processor(FeatureProcessor): Feature processor for user and movie features
        num_negatives(int): Number of negative samples per positive sample
    """
    def __init__(self, ratings, precomputed_candidates, feature_processor, num_negatives=4):
        self.ratings = ratings
        self.precomputed_candidates = precomputed_candidates
        self.feature_processor = feature_processor
        self.num_negatives = num_negatives
        self.user_features, self.movie_features, self.labels = self.get_dataset()
        
    def __len__(self):
        return len(self.user_features)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.movie_features[idx], self.labels[idx]
    
    def get_dataset(self):
        user_features_list, movie_features_list, labels = [], [], []
        
        print(f"Generating training dataset with precomputed candidates")
        
        user_positive_items = self.ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
        
        for user_id, positive_items in tqdm(user_positive_items.items(), desc="Processing users"):
            if user_id not in self.precomputed_candidates:
                continue
                
            user_feat = self.feature_processor.get_user_features(user_id)
            available_candidates = self.precomputed_candidates[user_id]
            
            # Add positive samples
            for item_id in positive_items:
                movie_feat = self.feature_processor.get_movie_features(item_id)
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(1)
            
            # Add negative samples using precomputed candidates
            self._add_negative_samples(user_features_list, movie_features_list, labels, user_id, user_feat, positive_items, available_candidates)
        
        print(f"Generated {len(user_features_list)} samples ({labels.count(1)} positive, {labels.count(0)} negative)")
        print(f"Negative-to-positive ratio: {labels.count(0) / labels.count(1):.2f}")
        
        # Convert to tensors
        user_features_tensor = torch.stack(user_features_list)
        movie_features_tensor = torch.stack(movie_features_list)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return user_features_tensor, movie_features_tensor, labels_tensor
    
    def _add_negative_samples(self, user_features_list, movie_features_list, labels, user_id, user_feat, positive_items, available_candidates):
        """Add negative samples using precomputed candidates"""
        total_negatives_needed = len(positive_items) * self.num_negatives
        
        # Filter out positive items from candidates
        negative_candidates = [item for item in available_candidates if item not in positive_items]
        
        # Sample negative items
        if len(negative_candidates) > 0:
            sample_size = min(total_negatives_needed, len(negative_candidates))
            sampled_negatives = np.random.choice(
                negative_candidates,
                size=sample_size,
                replace=False
            )
            
            # Add sampled negatives
            for neg_item in sampled_negatives:
                movie_feat = self.feature_processor.get_movie_features(neg_item)
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(0)