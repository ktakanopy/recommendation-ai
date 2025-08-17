import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from movie_lens_dataset import MovieLensTrainDataset

np.random.seed(123)


class NCF(nn.Module):
    """ Optimized Neural Collaborative Filtering (NCF) with Features and Smart Negative Sampling
    
        Args:
            user_feature_dim (int): Dimension of user features
            movie_feature_dim (int): Dimension of movie features
            ratings (pd.DataFrame): Dataframe containing the movie ratings for training
            feature_processor (FeatureProcessor): Feature processor for user and movie features
            candidate_generator (CandidateGenerator): Optimized candidate generator
            negative_method (str): Method for negative sampling
            sampling_strategy (str): Strategy for negative sampling
    """
    
    def __init__(self, user_feature_dim, movie_feature_dim, ratings, feature_processor, candidate_generator, 
                 negative_method="hybrid", sampling_strategy="unique_per_user"):
        super().__init__()
        
        # Store parameters
        self.user_feature_dim = user_feature_dim
        self.movie_feature_dim = movie_feature_dim
        self.ratings = ratings
        self.feature_processor = feature_processor
        self.candidate_generator = candidate_generator
        self.negative_method = negative_method
        self.sampling_strategy = sampling_strategy
        
        # Feature processing layers with batch normalization
        self.user_fc1 = nn.Linear(user_feature_dim, 128)
        self.user_bn1 = nn.BatchNorm1d(128)
        self.user_fc2 = nn.Linear(128, 64)
        self.user_bn2 = nn.BatchNorm1d(64)
        
        self.movie_fc1 = nn.Linear(movie_feature_dim, 256)
        self.movie_bn1 = nn.BatchNorm1d(256)
        self.movie_fc2 = nn.Linear(256, 128)
        self.movie_bn2 = nn.BatchNorm1d(128)
        self.movie_fc3 = nn.Linear(128, 64)
        self.movie_bn3 = nn.BatchNorm1d(64)
        
        # NCF layers
        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, user_features, movie_features):
        """
        Forward pass with user and movie feature vectors
        
        Args:
            user_features: Tensor of user features [batch_size, user_feature_dim]
            movie_features: Tensor of movie features [batch_size, movie_feature_dim]
        """
        
        # Process user features
        user_x = self.dropout(nn.ReLU()(self.user_bn1(self.user_fc1(user_features))))
        user_processed = nn.ReLU()(self.user_bn2(self.user_fc2(user_x)))
        
        # Process movie features
        movie_x = self.dropout(nn.ReLU()(self.movie_bn1(self.movie_fc1(movie_features))))
        movie_x = self.dropout(nn.ReLU()(self.movie_bn2(self.movie_fc2(movie_x))))
        movie_processed = nn.ReLU()(self.movie_bn3(self.movie_fc3(movie_x)))
        
        # Concat the processed features
        vector = torch.cat([user_processed, movie_processed], dim=-1)
        
        # Pass through NCF layers
        vector = self.dropout(nn.ReLU()(self.fc1(vector)))
        vector = self.dropout(nn.ReLU()(self.fc2(vector)))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred
    
    def compute_loss(self, batch):
        user_features, movie_features, labels = batch
        predicted_labels = self(user_features, movie_features)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def get_dataloader(self, batch_size=512, num_workers=4, num_negatives=4):
        """Get DataLoader with optimized negative sampling"""
        dataset = MovieLensTrainDataset(
            self.ratings, 
            self.candidate_generator,
            self.feature_processor,
            num_negatives=num_negatives,
            negative_method=self.negative_method,
            sampling_strategy=self.sampling_strategy
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    def save_weights(self, filepath):
        """
        Save model weights and configuration to file
        
        Args:
            filepath (str): Path to save the model weights
        """
        model_state = {
            'state_dict': self.state_dict(),
            'user_feature_dim': self.user_feature_dim,
            'movie_feature_dim': self.movie_feature_dim,
            'negative_method': self.negative_method,
            'sampling_strategy': self.sampling_strategy,
            'model_type': 'NCF'
        }
        torch.save(model_state, filepath)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath, strict=True):
        """
        Load model weights from file
        
        Args:
            filepath (str): Path to the saved model weights
            strict (bool): Whether to strictly enforce that the keys in state_dict 
                          match the keys returned by this module's state_dict()
        """
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Verify model compatibility
            if checkpoint.get('model_type') != 'NCF':
                print(f"Warning: Model type mismatch. Expected NCF, got {checkpoint.get('model_type')}")
            
            if checkpoint.get('user_feature_dim') != self.user_feature_dim:
                raise ValueError(f"User feature dimension mismatch. Expected {self.user_feature_dim}, "
                               f"got {checkpoint.get('user_feature_dim')}")
            
            if checkpoint.get('movie_feature_dim') != self.movie_feature_dim:
                raise ValueError(f"Movie feature dimension mismatch. Expected {self.movie_feature_dim}, "
                               f"got {checkpoint.get('movie_feature_dim')}")
            
            # Load the state dict
            self.load_state_dict(checkpoint['state_dict'], strict=strict)
            
            # Update sampling configuration if available
            if 'negative_method' in checkpoint:
                self.negative_method = checkpoint['negative_method']
            if 'sampling_strategy' in checkpoint:
                self.sampling_strategy = checkpoint['sampling_strategy']
            
            print(f"Model weights loaded successfully from {filepath}")
            print(f"Loaded configuration: negative_method={self.negative_method}, "
                  f"sampling_strategy={self.sampling_strategy}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {str(e)}")
    
    @classmethod
    def load_model(cls, filepath, ratings, feature_processor, candidate_generator):
        """
        Class method to load a complete model from saved weights
        
        Args:
            filepath (str): Path to the saved model weights
            ratings: Training ratings dataframe
            feature_processor: Feature processor instance
            candidate_generator: Candidate generator instance
            
        Returns:
            NCF: Loaded NCF model instance
        """
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Create model with saved configuration
            model = cls(
                user_feature_dim=checkpoint['user_feature_dim'],
                movie_feature_dim=checkpoint['movie_feature_dim'],
                ratings=ratings,
                feature_processor=feature_processor,
                candidate_generator=candidate_generator,
                negative_method=checkpoint.get('negative_method', 'hybrid'),
                sampling_strategy=checkpoint.get('sampling_strategy', 'unique_per_user')
            )
            
            # Load weights
            model.load_state_dict(checkpoint['state_dict'])
            
            print(f"Complete model loaded successfully from {filepath}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading complete model: {str(e)}")