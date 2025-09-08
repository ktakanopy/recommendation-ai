import argparse
import os
import pickle
import sys

import pandas as pd
import torch


from neural_recommendation.applications.use_cases.deep_learning.candidate_generator import CandidateGenerator
from trainer import NCFTrainer

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import (
    FeatureInfoDto,
    SentenceEmbeddingsDto,
)
from neural_recommendation.applications.use_cases.deep_learning.ncf_feature_processor import NCFFeatureProcessor as FeatureProcessor
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.infrastructure.config.settings import MLModelSettings

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_and_save_features(users_df, movies_df, device, output_path):
    processor = FeatureProcessor(debug=False)
    processor.prepare_user_features(users_df)
    movie_prep_features = processor.prepare_movie_features(movies_df, device=device)

    movie_embeddings = movie_prep_features["movie_embeddings"]
    title_to_idx = movie_prep_features["title_to_idx"]
    idx_to_title = movie_prep_features["idx_to_title"]
    movie_genres_dict = movie_prep_features["movies_genres_dict"]

    age_mean = float(users_df["age"].mean())
    age_std = float(users_df["age"].std()) or 1.0

    embedding_matrix = movie_embeddings.to(device)
    embedding_dim = int(embedding_matrix.shape[1])

    sentence_embeddings = SentenceEmbeddingsDto(
        embedding_matrix=embedding_matrix,
        embedding_dim=embedding_dim,
        title_to_idx=title_to_idx,
        idx_to_title=idx_to_title,
        movies_genres_dict=movie_genres_dict,
    )

    feature_info = FeatureInfoDto(
        age_mean=age_mean,
        age_std=age_std,
        sentence_embeddings=sentence_embeddings,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving feature info to {output_path}")
    feature_info.save(output_path)
    settings = MLModelSettings()
    processor.save(settings.feature_processor_path)
    return processor


def precompute_and_save_candidates(train_df, val_df, movies_df, out_train, out_val, num_candidates):
    all_movie_ids = movies_df["movie_id"].astype(int).tolist()
    gen = CandidateGenerator(train_ratings=train_df, movies=movies_df, all_movieIds=all_movie_ids)
    train_candidates = gen.precompute_training_candidates(train_df, method="hybrid", num_candidates=num_candidates)
    val_candidates = gen.precompute_validation_candidates(
        val_df, train_df, method="hybrid", num_candidates=num_candidates
    )
    os.makedirs(os.path.dirname(out_train), exist_ok=True)
    with open(out_train, "wb") as f:
        pickle.dump(train_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(os.path.dirname(out_val), exist_ok=True)
    with open(out_val, "wb") as f:
        pickle.dump(val_candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
    return train_candidates, val_candidates


def split_data(ratings_df):
    ratings_df["rank_latest"] = ratings_df.groupby(["user_id"])["timestamp"].rank(method="first", ascending=False)

    # Training: Take all ratings except the last 2 (penultimate and latest)
    train_ratings = ratings_df[~ratings_df["rank_latest"].isin([1, 2])]

    # Validation: Take the penultimate rating (rank = 2)
    validation_ratings = ratings_df[ratings_df["rank_latest"] == 2]

    # Test: Take the latest rating (rank = 1)
    test_ratings = ratings_df[ratings_df["rank_latest"] == 1]

    # drop columns that we no longer need
    train_ratings = train_ratings[["user_id", "movie_id", "rating"]]
    validation_ratings = validation_ratings[["user_id", "movie_id", "rating"]]  # Added this line
    test_ratings = test_ratings[["user_id", "movie_id", "rating"]]

    return train_ratings, validation_ratings, test_ratings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_out", type=str, default="data/processed_data/preprocessed_features.pkl")
    parser.add_argument(
        "--train_candidates_out",
        type=str,
        default="data/processed_data/precomputed_train_candidates.pkl",
    )
    parser.add_argument(
        "--val_candidates_out",
        type=str,
        default="data/processed_data/precomputed_val_candidates.pkl",
    )
    parser.add_argument("--model_out", type=str, default="models/ncf_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--num_candidates", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--use_existing_candidates",
        action="store_true",
        default=True,
        help="If set, load precomputed train/val candidates from the given output paths when they exist",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ratings_df = pd.read_csv(
        "data/ml-1m/ratings.dat",
        sep="::",
        header=None,
        engine="python",
    )
    ratings_df.columns = ["user_id", "movie_id", "rating", "timestamp"]

    movies_df = pd.read_csv(
        "data/ml-1m/movies.dat",
        sep="::",
        header=None,
        engine="python",
    )
    movies_df.columns = ["movie_id", "title", "genres"]

    users_df = pd.read_csv(
        "data/ml-1m/users.dat",
        sep="::",
        header=None,
        engine="python",
    )
    users_df.columns = ["user_id", "gender", "age", "occupation", "zip_code"]

    train_df, val_df, test_df = split_data(ratings_df)

    processor = compute_and_save_features(users_df, movies_df, device=str(device), output_path=args.features_out)

    if (
        args.use_existing_candidates
        and os.path.exists(args.train_candidates_out)
        and os.path.exists(args.val_candidates_out)
    ):
        with open(args.train_candidates_out, "rb") as f:
            train_candidates = pickle.load(f)
        with open(args.val_candidates_out, "rb") as f:
            val_candidates = pickle.load(f)
    else:
        train_candidates, val_candidates = precompute_and_save_candidates(
            train_df,
            val_df,
            movies_df,
            args.train_candidates_out,
            args.val_candidates_out,
            args.num_candidates,
        )

    model = NCFModel(
        user_feature_dim=int(processor.user_feature_dim),
        movie_feature_dim=int(processor.movie_feature_dim),
    ).to(device)

    trainer = NCFTrainer(
        model=model,
        train_ratings=train_df,
        train_candidates=train_candidates,
        validation_candidates=val_candidates,
        validation_ratings=val_df,
        feature_processor=processor,
        num_negatives=int(args.num_negatives),
        device=device,
    )

    trainer.train(
        num_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        k=int(args.k),
        verbose=True,
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    trainer.save_model(args.model_out, save_best=True)


if __name__ == "__main__":
    main()
