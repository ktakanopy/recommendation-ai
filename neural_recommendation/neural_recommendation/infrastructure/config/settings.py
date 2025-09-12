from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    DATABASE_URL: str
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int


class MLModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_prefix="ML_", extra="ignore")

    num_epochs: int = 10
    embedding_size: int = 64
    layer_sizes: List[int] = [32, 32, 32]
    device: str = "cpu"
    sample_users: int | None = None
    temperature: float = 0.1
    sentence_model: str = "all-MiniLM-L6-v2"
    models_dir: str = "models"
    model_name: str = "ncf.pth"
    processed_data_dir: str = "data/processed_data"
    feature_processor_path: str = "feature_processor.joblib"
    candidate_generator_path: str = "candidate_gen.pkl"
    user_features_index_path: str = "user_annoy.index"
    movie_features_index_path: str = "movie_annoy.index"
    movie_features_cache_path: str = "movie_features_cache.pkl"
    feature_encoder_index_path: str = "feature_encoder.pkl"
    top_popular_movies_path: str = "top_popular_movies.pkl"

    data_dir: str = "data/ml-1m"
    movie_name: str = "movies.dat"
    candidate_gen_name: str = "candidate_gen.pkl"
    history_name: str = "training_history.json"
    plot_name: str = "training_curves.png"
    retrain: bool = False

    batch_size: int = 64
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    loss_type: str = "sampled_softmax"
    num_negatives: int = 4
    num_sampled_negatives: int = 5000
    use_hard_negatives: bool = False
    hard_negative_ratio: float = 0.5

    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2

    # NCF-specific settings
    model_type: str = "ncf"  # "ncf" or "two_tower"
    user_feature_dim: int | None = None
    movie_feature_dim: int | None = None
