import numpy as np
import torch
from torch import nn

from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)

np.random.seed(123)


class NCFModel(nn.Module):
    """Neural Collaborative Filtering (NCF) with Features and Smart Negative Sampling

    Args:
        user_feature_dim (int): Dimension of user features
        movie_feature_dim (int): Dimension of movie features
        num_negatives (int): Number of negative samples per positive sample
    """

    def __init__(self, user_feature_dim: int, movie_feature_dim: int, num_negatives: int = 4):
        super().__init__()

        # Store parameters
        self.user_feature_dim = user_feature_dim
        self.movie_feature_dim = movie_feature_dim
        self.num_negatives = num_negatives

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

        logger.info(f"NCFModel initialized with user_dim={user_feature_dim}, movie_dim={movie_feature_dim}")

    def forward(self, user_features: torch.Tensor, movie_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with user and movie feature vectors

        Args:
            user_features: Tensor of user features [batch_size, user_feature_dim]
            movie_features: Tensor of movie features [batch_size, movie_feature_dim]

        Returns:
            Predicted interaction probability [batch_size, 1]
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

    def compute_loss(
        self, user_features: torch.Tensor, movie_features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss"""
        predicted_labels = self(user_features, movie_features)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def predict_batch(self, user_features: torch.Tensor, movie_features: torch.Tensor) -> torch.Tensor:
        """Predict interaction probabilities for a batch"""
        self.eval()
        with torch.no_grad():
            predictions = self(user_features, movie_features)
        return predictions.squeeze()

    def save_weights(self, filepath: str):
        """Save model weights and configuration to file"""
        model_state = {
            "state_dict": self.state_dict(),
            "user_feature_dim": self.user_feature_dim,
            "movie_feature_dim": self.movie_feature_dim,
            "num_negatives": self.num_negatives,
            "model_type": "NCF",
        }
        torch.save(model_state, filepath)
        logger.info(f"NCF model weights saved to {filepath}")

    def load_weights(self, filepath: str, strict: bool = True):
        """Load model weights from file"""
        try:
            checkpoint = torch.load(filepath, map_location="cpu")

            # Verify model compatibility
            if checkpoint.get("model_type") != "NCF":
                logger.warning(f"Model type mismatch. Expected NCF, got {checkpoint.get('model_type')}")

            if checkpoint.get("user_feature_dim") != self.user_feature_dim:
                raise ValueError(
                    f"User feature dimension mismatch. Expected {self.user_feature_dim}, "
                    f"got {checkpoint.get('user_feature_dim')}"
                )

            if checkpoint.get("movie_feature_dim") != self.movie_feature_dim:
                raise ValueError(
                    f"Movie feature dimension mismatch. Expected {self.movie_feature_dim}, "
                    f"got {checkpoint.get('movie_feature_dim')}"
                )

            # Load the state dict
            self.load_state_dict(checkpoint["state_dict"], strict=strict)

            logger.info(f"NCF model weights loaded successfully from {filepath}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Model weights file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading model weights: {str(e)}")

    @classmethod
    def load_model(cls, filepath: str, num_negatives: int = 4) -> "NCFModel":
        """Class method to load a complete model from saved weights"""
        try:
            checkpoint = torch.load(filepath, map_location="cpu")

            # Create model with saved configuration
            model = cls(
                user_feature_dim=checkpoint["user_feature_dim"],
                movie_feature_dim=checkpoint["movie_feature_dim"],
                num_negatives=num_negatives,
            )

            # Load weights
            model.load_state_dict(checkpoint["state_dict"])

            logger.info(f"Complete NCF model loaded successfully from {filepath}")
            return model

        except Exception as e:
            raise RuntimeError(f"Error loading complete NCF model: {str(e)}")
