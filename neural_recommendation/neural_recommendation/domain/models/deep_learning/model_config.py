from pydantic import BaseModel
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from neural_recommendation.infrastructure.config.settings import MLModelSettings


class ModelConfig(BaseModel):
    # Required fields (no defaults) - must come first
    num_epochs: int
    embedding_size: int
    layer_sizes: List[int]
    device: str
    sample_users: Optional[int]
    temperature: float
    sentence_model: str
    models_dir: str
    model_name: str
    history_name: str
    plot_name: str
    retrain: bool
    batch_size: int
    learning_rate: float
    max_grad_norm: float
    train_split: float
    val_split: float
    test_split: float
    loss_type: str
    num_sampled_negatives: int
    use_hard_negatives: bool
    hard_negative_ratio: float
    dropout_rate: float

    # Optional fields with defaults (NCF-specific parameters) - must come last
    user_feature_dim: Optional[int] = None
    movie_feature_dim: Optional[int] = None
    num_negatives: int = 4
    model_type: str = "ncf"  # "ncf" or "two_tower"

    @classmethod
    def from_settings(cls, settings: "MLModelSettings") -> "ModelConfig":
        return cls(
            num_epochs=settings.num_epochs,
            embedding_size=settings.embedding_size,
            layer_sizes=settings.layer_sizes,
            device=settings.device,
            sample_users=settings.sample_users,
            temperature=settings.temperature,
            sentence_model=settings.sentence_model,
            models_dir=settings.models_dir,
            model_name=settings.model_name,
            history_name=settings.history_name,
            plot_name=settings.plot_name,
            retrain=settings.retrain,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            max_grad_norm=settings.max_grad_norm,
            train_split=settings.train_split,
            val_split=settings.val_split,
            test_split=settings.test_split,
            num_negatives=getattr(settings, "num_negatives", 4),
            use_hard_negatives=getattr(settings, "use_hard_negatives", False),
            hard_negative_ratio=getattr(settings, "hard_negative_ratio", 0.5),
            loss_type=getattr(settings, "loss_type", "explicit_negatives"),
            num_sampled_negatives=getattr(settings, "num_sampled_negatives", 1000),
            dropout_rate=getattr(settings, "dropout_rate", 0.3),
            # NCF-specific settings
            model_type=getattr(settings, "model_type", "ncf"),
            user_feature_dim=getattr(settings, "user_feature_dim", None),
            movie_feature_dim=getattr(settings, "movie_feature_dim", None),
        )

    def to_dict(self) -> dict:
        return {
            "num_epochs": self.num_epochs,
            "embedding_size": self.embedding_size,
            "layer_sizes": self.layer_sizes,
            "device": self.device,
            "sample_users": self.sample_users,
            "temperature": self.temperature,
            "sentence_model": self.sentence_model,
            "models_dir": self.models_dir,
            "model_name": self.model_name,
            "history_name": self.history_name,
            "plot_name": self.plot_name,
            "retrain": self.retrain,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "loss_type": self.loss_type,
            "num_sampled_negatives": self.num_sampled_negatives,
            "use_hard_negatives": self.use_hard_negatives,
            "hard_negative_ratio": self.hard_negative_ratio,
            "dropout_rate": self.dropout_rate,
            "user_feature_dim": self.user_feature_dim,
            "movie_feature_dim": self.movie_feature_dim,
            "num_negatives": self.num_negatives,
            "model_type": self.model_type,
        }
