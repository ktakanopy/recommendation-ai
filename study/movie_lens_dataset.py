

from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm

class MovieLensTrainDataset(Dataset):
    """Optimized MovieLens Dataset using Candidate Generator for Negative Sampling with Features
    Args:
        ratings(pd.DataFrame): Dataframe containing the movie ratings
        candidate_generator(CandidateGenerator): Optimized candidate generator
        feature_processor(FeatureProcessor): Feature processor for user and movie features
        num_negatives(int): Number of negative samples per positive sample
        negative_method(str): Method for generating negative candidates
        sampling_strategy(str): Strategy for negative sampling ('unique_per_user', 'per_positive', 'stratified')
    """
    def __init__(self, ratings, candidate_generator, feature_processor, num_negatives=4, negative_method="hybrid", sampling_strategy="unique_per_user"):
        self.ratings = ratings
        self.candidate_generator = candidate_generator
        self.feature_processor = feature_processor
        self.num_negatives = num_negatives
        self.negative_method = negative_method
        self.sampling_strategy = sampling_strategy
        self.user_features, self.movie_features, self.labels = self.get_dataset()
        
    def __len__(self):
        return len(self.user_features)
    
    def __getitem__(self, idx):
        return self.user_features[idx], self.movie_features[idx], self.labels[idx]
    
    def get_dataset(self):
        user_features_list, movie_features_list, labels = [], [], []
        
        print(f"Generating training dataset with {self.negative_method} negative sampling")
        print(f"Sampling strategy: {self.sampling_strategy}")
        
        # Process each user to generate negatives using candidate generator
        user_positive_items = self.ratings.groupby('user_id')['movie_id'].apply(list).to_dict()
        
        for user_id, positive_items in tqdm(user_positive_items.items(), desc="Processing users"):
            # Get user features once for this user
            user_feat = self.feature_processor.get_user_features(user_id)
            
            # Add positive samples
            for item_id in positive_items:
                movie_feat = self.feature_processor.get_movie_features(item_id)
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(1)
            
            # Generate negatives based on sampling strategy
            if self.sampling_strategy == "unique_per_user":
                self._add_unique_negatives_per_user(user_features_list, movie_features_list, labels, user_id, user_feat, positive_items)
            elif self.sampling_strategy == "per_positive":
                self._add_negatives_per_positive(user_features_list, movie_features_list, labels, user_id, user_feat, positive_items)
            elif self.sampling_strategy == "stratified":
                self._add_stratified_negatives(user_features_list, movie_features_list, labels, user_id, user_feat, positive_items)
            else:
                raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
        
        print(f"Generated {len(user_features_list)} samples ({labels.count(1)} positive, {labels.count(0)} negative)")
        print(f"Negative-to-positive ratio: {labels.count(0) / labels.count(1):.2f}")
        
        # Convert to tensors
        user_features_tensor = torch.stack(user_features_list)
        movie_features_tensor = torch.stack(movie_features_list)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return user_features_tensor, movie_features_tensor, labels_tensor
    
    def _add_unique_negatives_per_user(self, user_features_list, movie_features_list, labels, user_id, user_feat, positive_items):
        """Add unique negative samples per user (most efficient approach)"""
        total_negatives_needed = len(positive_items) * self.num_negatives
        
        # Generate candidate pool
        negative_candidates = self.candidate_generator.generate_candidates(
            user_id, 
            method=self.negative_method, 
            num_candidates=min(total_negatives_needed * 2, 300)  # Get more candidates than needed
        )
        
        # Filter out positive items to ensure no overlap
        negative_candidates = [item for item in negative_candidates if item not in positive_items]
        
        # Sample unique negatives
        if len(negative_candidates) > 0:
            sample_size = min(total_negatives_needed, len(negative_candidates))
            sampled_negatives = np.random.choice(
                negative_candidates,
                size=sample_size,
                replace=False
            )
            
            # Add all sampled negatives for this user
            for neg_item in sampled_negatives:
                movie_feat = self.feature_processor.get_movie_features(neg_item)
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(0)
    
    def _add_negatives_per_positive(self, user_features_list, movie_features_list, labels, user_id, user_feat, positive_items):
        """Add negatives per positive item (original intended approach but fixed)"""
        # Generate candidate pool once per user
        negative_candidates = self.candidate_generator.generate_candidates(
            user_id,
            method=self.negative_method,
            num_candidates=min(len(positive_items) * self.num_negatives * 2, 300)
        )
        
        # Filter out positive items
        negative_candidates = [item for item in negative_candidates if item not in positive_items]
        
        if len(negative_candidates) == 0:
            return
        
        # For each positive item, sample unique negatives
        all_used_negatives = set()
        for _ in positive_items:
            # Available negatives (excluding already used ones)
            available_negatives = [item for item in negative_candidates if item not in all_used_negatives]
            
            if len(available_negatives) == 0:
                break
                
            # Sample negatives for this positive item
            sample_size = min(self.num_negatives, len(available_negatives))
            sampled_negatives = np.random.choice(
                available_negatives,
                size=sample_size,
                replace=False
            )
            
            # Add negatives and mark as used
            for neg_item in sampled_negatives:
                movie_feat = self.feature_processor.get_movie_features(neg_item)
                user_features_list.append(user_feat)
                movie_features_list.append(movie_feat)
                labels.append(0)
                all_used_negatives.add(neg_item)
    
    def _add_stratified_negatives(self, user_features_list, movie_features_list, labels, user_id, user_feat, positive_items):
        """Add stratified negatives from different candidate methods"""
        total_negatives_needed = len(positive_items) * self.num_negatives
        
        # Get negatives from different strategies
        methods = ["popularity", "collaborative", "content"]
        negatives_per_method = total_negatives_needed // len(methods)
        
        all_negatives = []
        
        for method in methods:
            method_candidates = self.candidate_generator.generate_candidates(
                user_id,
                method=method,
                num_candidates=negatives_per_method * 2
            )
            # Filter out positives and already selected negatives
            method_candidates = [item for item in method_candidates 
                               if item not in positive_items and item not in all_negatives]
            
            # Sample from this method
            if len(method_candidates) > 0:
                sample_size = min(negatives_per_method, len(method_candidates))
                sampled = np.random.choice(method_candidates, size=sample_size, replace=False)
                all_negatives.extend(sampled)
        
        # Fill remaining slots with random negatives if needed
        remaining_slots = total_negatives_needed - len(all_negatives)
        if remaining_slots > 0:
            available_items = self.candidate_generator.get_available_items(user_id)
            remaining_items = [item for item in available_items if item not in all_negatives]
            
            if len(remaining_items) > 0:
                sample_size = min(remaining_slots, len(remaining_items))
                additional_negatives = np.random.choice(remaining_items, size=sample_size, replace=False)
                all_negatives.extend(additional_negatives)
        
        # Add all negatives for this user
        for neg_item in all_negatives:
            movie_feat = self.feature_processor.get_movie_features(neg_item)
            user_features_list.append(user_feat)
            movie_features_list.append(movie_feat)
            labels.append(0)