import torch
import torch.nn.functional as F
from collections import Counter

class ColdStartRecommender:
    """
    Cold Start Recommendation System for new users
    
    Handles recommendations for users with limited or no interaction history
    by leveraging user demographics, initial ratings, and content-based filtering.
    """
    
    def __init__(self, trained_model, feature_processor, candidate_generator, movies_df, liked_threshold=4):
        self.model = trained_model
        self.feature_processor = feature_processor
        self.candidate_generator = candidate_generator
        self.movies_df = movies_df
        self.device = next(trained_model.parameters()).device
        self.liked_threshold = liked_threshold
        

    
    def get_similar_users_neural_embedding(self, user_demographics, top_k=50):
        """
        Find users with similar demographics using neural network embeddings
        
        Uses the trained NCF model's user feature processing layers to compute
        user embeddings and find similar users based on cosine similarity.
        
        Args:
            user_demographics: dict with user demographic info
            top_k: number of similar users to return
            
        Returns:
            list: user_ids of similar users
        """
        
        # Get the new user's feature vector and neural representation
        new_user_features = self.feature_processor.process_user_demographics(user_demographics)
        new_user_features = new_user_features.detach().clone().unsqueeze(0).to(self.device)
        
        # Get neural representation using the trained model's user processing layers
        self.model.eval()
        with torch.no_grad():
            # Process through the trained neural network layers (same as in NCF forward pass)
            user_x = self.model.dropout(torch.relu(self.model.user_bn1(self.model.user_fc1(new_user_features))))
            new_user_embedding = torch.relu(self.model.user_bn2(self.model.user_fc2(user_x)))
        
        # Get embeddings for all existing users
        similar_users = []
        existing_user_ids = list(self.feature_processor.user_features_cache.keys())
        
        # Process existing users in batches for efficiency
        batch_size = 128
        for i in range(0, len(existing_user_ids), batch_size):
            batch_user_ids = existing_user_ids[i:i + batch_size]
            batch_features = []
            
            for user_id in batch_user_ids:
                cached_features = self.feature_processor.user_features_cache[user_id]
                batch_features.append(cached_features)
            
            # Convert to tensor and move to device
            batch_features = torch.stack(batch_features).to(self.device)
            
            # Get neural embeddings for this batch
            with torch.no_grad():
                batch_x = self.model.dropout(torch.relu(self.model.user_bn1(self.model.user_fc1(batch_features))))
                batch_embeddings = torch.relu(self.model.user_bn2(self.model.user_fc2(batch_x)))
                
                # Compute cosine similarity with the new user
                similarities = F.cosine_similarity(new_user_embedding, batch_embeddings, dim=1)
                
                # Store results
                for j, user_id in enumerate(batch_user_ids):
                    similarity_score = similarities[j].item()
                    similar_users.append((user_id, similarity_score))
        
        # Sort by similarity score (descending) and return top_k
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in similar_users[:top_k]]

    def get_similar_user_candidates(self, user_demographics, top_k=10, num_candidates=100):
        similar_users = self.get_similar_users_neural_embedding(user_demographics)
        candidates = []
        if similar_users:
            similar_user_movies = []
            for similar_user_id in similar_users[:top_k]:  # Top 10 similar users
                user_movies = self.candidate_generator.user_interacted_items.get(similar_user_id, [])
                similar_user_movies.extend(user_movies)
            
            movie_counts = Counter(similar_user_movies)
            demographic_candidates = [movie_id for movie_id, _ in movie_counts.most_common(num_candidates//2)]
            candidates.extend(demographic_candidates)
        return candidates


    def generate_cold_start_candidates(self, user_demographics, user_ratings=None, num_candidates=100):
        """
        Generate candidate movies for cold start scenario
        
        Args:
            user_demographics: dict with user demographic info
            user_ratings: list of (movie_id, rating) tuples for initial ratings
            num_candidates: number of candidates to generate
            
        Returns:
            list: candidate movie IDs
        """
        candidates = []
        
        if user_ratings is None or len(user_ratings) == 0:
            # Pure cold start - no ratings yet
            # Use hybrid (popularity + collaborative + content) + demographic-based recommendations
            
            # Get hybrid candidates (popularity + collaborative + content)
            hybrid_candidates = self.candidate_generator.generate_hybrid_candidates(
                user_id=-1,  # dummy user_id
                num_candidates=num_candidates//2
            )
            candidates.extend(hybrid_candidates)
            
            # Get recommendations based on similar users' preferences (using neural embeddings)
            similar_users = self.get_similar_users_neural_embedding(user_demographics)
            if similar_users:
                similar_user_movies = []
                for similar_user_id in similar_users[:10]:  # Top 10 similar users
                    user_movies = self.candidate_generator.user_interacted_items.get(similar_user_id, [])
                    similar_user_movies.extend(user_movies)
                
                movie_counts = Counter(similar_user_movies)
                demographic_candidates = [movie_id for movie_id, _ in movie_counts.most_common(num_candidates//2)]
                candidates.extend(demographic_candidates)
        
        else:
            liked_movies = [movie_id for movie_id, rating in user_ratings if rating >= self.liked_threshold]
            user_similar_candidates = self.get_similar_user_candidates(user_demographics, top_k=10, num_candidates=num_candidates//3)
            popular_candidates = self.candidate_generator.generate_popularity_candidates(user_id=-1, num_candidates=num_candidates//3, user_available_items=liked_movies)
            user_genres = self.candidate_generator.get_genres_from_movies(liked_movies)
            content_candidates = self.candidate_generator.generate_content_candidates(user_id=-1, num_candidates=num_candidates//3, user_available_items=liked_movies, passed_user_genres=user_genres)
            candidates.extend(user_similar_candidates)
            candidates.extend(popular_candidates)
            candidates.extend(content_candidates)
            
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for movie_id in candidates:
            if movie_id not in seen:
                seen.add(movie_id)
                unique_candidates.append(movie_id)
        
        return unique_candidates[:num_candidates]
    
    def recommend_for_new_user(self, user_demographics, user_ratings=None, num_recommendations=10):
        """
        Generate recommendations for a new user
        
        Args:
            user_demographics: dict with keys 'gender', 'age', 'occupation'
            user_ratings: list of (movie_id, rating) tuples for initial ratings (optional)
            num_recommendations: number of recommendations to return
            
        Returns:
            list: list of (movie_id, title, predicted_score) tuples
        """
        # Create user feature vector using FeatureProcessor
        user_features = self.feature_processor.process_user_demographics(user_demographics)
        user_features = user_features.detach().clone().unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Generate candidate movies
        candidates = self.generate_cold_start_candidates(
            user_demographics, 
            user_ratings, 
            num_candidates=min(200, len(self.movies_df))
        )
        
        # Score candidates using the NCF model
        movie_scores = []
        
        self.model.eval()
        with torch.no_grad():
            for movie_id in candidates:
                if movie_id in self.feature_processor.movie_features_cache:
                    # Get movie features
                    movie_features = self.feature_processor.get_movie_features(movie_id)
                    movie_features = movie_features.unsqueeze(0).to(self.device)
                    
                    # Predict score using NCF model
                    score = self.model(user_features, movie_features).item()
                    
                    # Get movie title
                    movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].iloc[0]
                    
                    movie_scores.append((movie_id, movie_title, score))
        
        # Sort by predicted score and return top recommendations
        movie_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Filter out movies user has already rated
        if user_ratings:
            rated_movie_ids = {movie_id for movie_id, _ in user_ratings}
            movie_scores = [(mid, title, score) for mid, title, score in movie_scores 
                           if mid not in rated_movie_ids]
        
        return movie_scores[:num_recommendations]
    
    def get_onboarding_movies(self, num_movies=10):
        """
        Get diverse, popular movies for new user onboarding/rating collection
        
        Args:
            num_movies: number of movies to return for rating
            
        Returns:
            list: list of (movie_id, title, genres) tuples
        """
        # Get hybrid candidates from different genres for diversity
        popular_movies = self.candidate_generator.generate_hybrid_candidates(
            user_id=-1, 
            num_candidates=100
        )
        
        # Group by genres to ensure diversity
        genre_movies = {}
        selected_movies = []
        
        for movie_id in popular_movies:
            if movie_id in self.candidate_generator.movie_to_genres:
                movie_genres = self.candidate_generator.movie_to_genres[movie_id]
                movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].iloc[0]
                movie_genres_str = self.movies_df[self.movies_df['movie_id'] == movie_id]['genres'].iloc[0]
                
                # Add to genre groups
                for genre in movie_genres:
                    if genre not in genre_movies:
                        genre_movies[genre] = []
                    genre_movies[genre].append((movie_id, movie_title, movie_genres_str))
        
        # Select diverse movies (one from each genre initially)
        used_genres = set()
        used_movie_ids = set()
        for genre, movies in genre_movies.items():
            if len(selected_movies) < num_movies and genre not in used_genres:
                # Find the first movie in this genre that hasn't been selected yet
                for movie_tuple in movies:
                    movie_id = movie_tuple[0]
                    if movie_id not in used_movie_ids:
                        selected_movies.append(movie_tuple)
                        used_genres.add(genre)
                        used_movie_ids.add(movie_id)
                        break
        
        # Fill remaining slots with most popular movies
        for movie_id in popular_movies:
            if len(selected_movies) >= num_movies:
                break
            
            if movie_id not in used_movie_ids:
                movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].iloc[0]
                movie_genres_str = self.movies_df[self.movies_df['movie_id'] == movie_id]['genres'].iloc[0]
                
                movie_tuple = (movie_id, movie_title, movie_genres_str)
                selected_movies.append(movie_tuple)
                used_movie_ids.add(movie_id)
        
        return selected_movies[:num_movies]
