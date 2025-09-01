from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Feature preprocessing functions
class FeatureProcessor:
    def __init__(self, debug=False):
        self.user_encoders = {}
        self.movie_encoders = {}
        self.sentence_model = None
        self.user_features_cache = {}
        self.movie_features_cache = {}
        self.debug = debug

        # sklearn encoders for user features
        self.gender_encoder = LabelEncoder()
        self.age_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.occupation_encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.encoders_fitted = False

    def prepare_user_features(self, users_df):
        """Prepare user features: gender, age, occupation one-hot encoding"""
        if self.debug:
            print("Preparing user features...")

        # Fit and transform using sklearn encoders
        if self.debug:
            print("Fitting sklearn encoders...")

        # Gender encoding (M=1, F=0) using LabelEncoder
        gender_encoded = self.gender_encoder.fit_transform(
            users_df["gender"].values
        ).astype(float)

        # Age one-hot encoding using OneHotEncoder
        age_onehot = self.age_encoder.fit_transform(
            users_df["age"].values.reshape(-1, 1)
        )

        # Occupation one-hot encoding using OneHotEncoder
        occupation_onehot = self.occupation_encoder.fit_transform(
            users_df["occupation"].values.reshape(-1, 1)
        )

        # Mark encoders as fitted
        self.encoders_fitted = True

        if self.debug:
            print(f"Gender categories: {self.gender_encoder.classes_}")
            print(f"Age categories: {self.age_encoder.categories_[0]}")
            print(f"Occupation categories: {self.occupation_encoder.categories_[0]}")

        # Convert to DataFrames for consistency with original format
        age_onehot_df = pd.DataFrame(
            age_onehot,
            columns=[f"age_{cat}" for cat in self.age_encoder.categories_[0]],
        )
        occupation_onehot_df = pd.DataFrame(
            occupation_onehot,
            columns=[f"occ_{cat}" for cat in self.occupation_encoder.categories_[0]],
        )

        # Combine all user features into a single DataFrame
        feature_columns = ["user_id"]
        feature_data = [users_df["user_id"].values]

        # Add gender
        feature_columns.append("gender")
        feature_data.append(gender_encoded)

        # Add age features
        for col in age_onehot_df.columns:
            feature_columns.append(col)
            feature_data.append(age_onehot_df[col].values)

        # Add occupation features
        for col in occupation_onehot_df.columns:
            feature_columns.append(col)
            feature_data.append(occupation_onehot_df[col].values)

        # Create feature matrix
        feature_matrix = np.column_stack(feature_data)
        user_features = pd.DataFrame(feature_matrix, columns=feature_columns)

        # Ensure all feature columns (except user_id) are float
        for col in user_features.columns:
            if col != "user_id":
                user_features[col] = user_features[col].astype(float)

        if self.debug:
            print(f"User features shape: {user_features.shape}")
            print(f"User feature columns: {list(user_features.columns)}")
            print(f"User feature dtypes:\n{user_features.dtypes}")

        # Cache features for quick lookup (keep on CPU)
        for _, row in user_features.iterrows():
            user_id = int(row["user_id"])
            # Get feature values excluding user_id and convert to numpy array
            feature_values = row.drop("user_id").values.astype(np.float32)
            features = torch.tensor(feature_values, dtype=torch.float32)
            self.user_features_cache[user_id] = features

        self.user_feature_dim = len(user_features.columns) - 1  # Exclude user_id
        print(f"User feature dimension: {self.user_feature_dim}")
        return user_features

    def prepare_movie_features(self, movies_df, device="cpu"):
        """Prepare movie features using sentence transformers for title and genres"""
        if self.debug:
            print("Preparing movie features with sentence transformers...")

        # Initialize sentence transformer
        if self.sentence_model is None:
            if self.debug:
                print("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        # Encode movie titles
        print("Encoding movie titles...")
        titles = movies_df["title"].tolist()
        title_embeddings = self.sentence_model.encode(
            titles, convert_to_tensor=True, device=device, batch_size=64
        )

        # Process genres and encode them
        print("Encoding movie genres...")
        genre_texts = []
        for genres_str in movies_df["genres"]:
            # Convert pipe-separated genres to readable text
            genres_list = genres_str.split("|")
            genre_text = " ".join(genres_list).replace("Children's", "Children")
            genre_texts.append(genre_text)

        genre_embeddings = self.sentence_model.encode(
            genre_texts, convert_to_tensor=True, device=device, batch_size=64
        )

        # Concatenate title and genre embeddings
        movie_embeddings = torch.cat([title_embeddings, genre_embeddings], dim=1)

        if self.debug:
            print(f"Title embeddings shape: {title_embeddings.shape}")
            print(f"Genre embeddings shape: {genre_embeddings.shape}")
            print(f"Combined movie embeddings shape: {movie_embeddings.shape}")

        # Normalize the embeddings for better training stability
        print("Normalizing movie embeddings...")
        movie_embeddings_normalized = F.normalize(movie_embeddings, p=2, dim=1)

        # Print normalization stats
        if self.debug:
            print(
                f"Original embeddings - mean: {movie_embeddings.mean().item():.4f}, std: {movie_embeddings.std().item():.4f}"
            )
            print(
                f"Normalized embeddings - mean: {movie_embeddings_normalized.mean().item():.4f}, std: {movie_embeddings_normalized.std().item():.4f}"
            )

        # Cache features for quick lookup (move to CPU for storage)
        for idx, row in movies_df.iterrows():
            movie_id = int(row["movie_id"])
            # Store normalized embeddings on CPU to avoid memory issues
            self.movie_features_cache[movie_id] = movie_embeddings_normalized[idx].cpu()

        self.movie_feature_dim = movie_embeddings_normalized.shape[1]
        if self.debug:
            print(f"Movie feature dimension: {self.movie_feature_dim}")
        return movie_embeddings_normalized

    def get_user_features(self, user_id):
        """Get cached user features"""
        return self.user_features_cache.get(user_id, torch.zeros(self.user_feature_dim))

    def get_movie_features(self, movie_id):
        """Get cached movie features"""
        return self.movie_features_cache.get(
            movie_id, torch.zeros(self.movie_feature_dim)
        )

    def process_user_demographics(self, user_demographics):
        """
        Process user demographics into feature vector format using fitted sklearn encoders

        Args:
            user_demographics: dict with keys: 'gender', 'age', 'occupation'
                - gender: 'M' or 'F'
                - age: int (age category from training data)
                - occupation: int (occupation code from training data)

        Returns:
            torch.Tensor: User feature vector (same format as cached features)
        """
        if not self.encoders_fitted:
            raise ValueError("Encoders not fitted. Call prepare_user_features first.")

        # Gender encoding using fitted LabelEncoder
        gender_encoded = self.gender_encoder.transform([user_demographics["gender"]])[
            0
        ].astype(float)

        # Age one-hot encoding using fitted OneHotEncoder
        age_onehot = self.age_encoder.transform([[user_demographics["age"]]])[0]

        # Occupation one-hot encoding using fitted OneHotEncoder
        occupation_onehot = self.occupation_encoder.transform(
            [[user_demographics["occupation"]]]
        )[0]

        # Combine all features (same order as prepare_user_features)
        feature_vector = np.concatenate(
            [[gender_encoded], age_onehot, occupation_onehot]
        )

        return torch.tensor(feature_vector, dtype=torch.float32)

    def get_encoder_info(self):
        """
        Get information about the fitted encoders

        Returns:
            dict: Information about encoders and their categories
        """
        if not self.encoders_fitted:  # TODO: remove this gambiarra
            return {"error": "Encoders not fitted yet"}

        return {
            "gender_classes": self.gender_encoder.classes_.tolist(),
            "age_categories": self.age_encoder.categories_[0].tolist(),
            "occupation_categories": self.occupation_encoder.categories_[0].tolist(),
            "total_features": 1
            + len(self.age_encoder.categories_[0])
            + len(self.occupation_encoder.categories_[0]),
        }
