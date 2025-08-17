import torch

class ColdStartRecommender:
    """
    Cold Start Recommendation System for new users
    
    Handles recommendations for users with limited or no interaction history
    by leveraging user demographics, initial ratings, and content-based filtering.
    """
    
    def __init__(self, trained_model, feature_processor, candidate_generator, movies_df):
        self.model = trained_model
        self.feature_processor = feature_processor
        self.candidate_generator = candidate_generator
        self.movies_df = movies_df
        self.device = next(trained_model.parameters()).device
        
    def create_user_features(self, user_demographics):
        """
        Create user feature vector from demographics
        
        Args:
            user_demographics: dict with keys: 'gender', 'age', 'occupation'
                - gender: 'M' or 'F'
                - age: int (1, 18, 25, 35, 45, 50, 56)
                - occupation: int (0-20)
        
        Returns:
            torch.Tensor: User feature vector
        """
        gender_encoded = 1.0 if user_demographics['gender'] == 'M' else 0.0
        
        # Create age one-hot (7 categories)
        age_categories = [1, 18, 25, 35, 45, 50, 56]
        age_onehot = [1.0 if user_demographics['age'] == cat else 0.0 for cat in age_categories]
        
        # Create occupation one-hot (21 categories: 0-20)
        occupation_onehot = [1.0 if user_demographics['occupation'] == i else 0.0 for i in range(21)]
        
        # Combine all features
        feature_vector = [gender_encoded] + age_onehot + occupation_onehot
        
        return torch.tensor(feature_vector, dtype=torch.float32)
    
    def get_similar_users_by_demographics(self, user_demographics, top_k=50):
        """
        Find users with similar demographics for collaborative filtering
        
        Args:
            user_demographics: dict with user demographic info
            top_k: number of similar users to return
            
        Returns:
            list: user_ids of similar users
        """
        similar_users = []
        
        # Simple demographic matching - can be made more sophisticated
        for user_id, cached_features in self.feature_processor.user_features_cache.items():
            # Check gender match (first feature)
            gender_match = (cached_features[0].item() == (1.0 if user_demographics['gender'] == 'M' else 0.0))
            
            # Check age category match (positions 1-7)
            age_categories = [1, 18, 25, 35, 45, 50, 56]
            user_age_idx = age_categories.index(user_demographics['age']) if user_demographics['age'] in age_categories else -1
            if user_age_idx >= 0:
                age_match = cached_features[1 + user_age_idx].item() == 1.0
            else:
                age_match = False
            
            # Check occupation match (positions 8-28)
            occ_match = False
            if 0 <= user_demographics['occupation'] <= 20:
                occ_match = cached_features[8 + user_demographics['occupation']].item() == 1.0
            
            # Score based on matches (prioritize age and occupation)
            score = 0
            if gender_match: score += 1
            if age_match: score += 2
            if occ_match: score += 3
            
            if score >= 2:  # Require at least age or occupation match
                similar_users.append((user_id, score))
        
        # Sort by score and return top_k
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in similar_users[:top_k]]
    
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
            # Use popularity + demographic-based recommendations
            
            # Get popular movies
            popular_candidates = self.candidate_generator.generate_popularity_candidates(
                user_id=-1,  # dummy user_id
                num_candidates=num_candidates//2
            )
            candidates.extend(popular_candidates)
            
            # Get recommendations based on similar users' preferences
            similar_users = self.get_similar_users_by_demographics(user_demographics)
            if similar_users:
                # Get popular movies among similar users
                similar_user_movies = []
                for similar_user_id in similar_users[:10]:  # Top 10 similar users
                    user_movies = self.candidate_generator.user_interacted_items.get(similar_user_id, [])
                    similar_user_movies.extend(user_movies)
                
                # Count frequency and get most popular among similar users
                from collections import Counter
                movie_counts = Counter(similar_user_movies)
                demographic_candidates = [movie_id for movie_id, _ in movie_counts.most_common(num_candidates//2)]
                candidates.extend(demographic_candidates)
        
        else:
            # Warm cold start - user has some initial ratings
            # Use content-based recommendations based on liked movies
            
            liked_movies = [movie_id for movie_id, rating in user_ratings if rating >= 4]
            
            if liked_movies:
                # Content-based recommendations using movie genres
                content_candidates = []
                
                # Get genres of liked movies
                liked_genres = []
                for movie_id in liked_movies:
                    if movie_id in self.candidate_generator.movie_to_genres:
                        liked_genres.extend(self.candidate_generator.movie_to_genres[movie_id])
                
                # Count genre preferences
                from collections import Counter
                genre_preferences = Counter(liked_genres)
                
                # Find movies with similar genres
                for movie_id, genres in self.candidate_generator.movie_to_genres.items():
                    if movie_id not in liked_movies:  # Don't recommend already rated movies
                        score = sum(genre_preferences.get(genre, 0) for genre in genres)
                        if score > 0:
                            content_candidates.append((movie_id, score))
                
                # Sort by content score and take top candidates
                content_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates.extend([movie_id for movie_id, _ in content_candidates[:num_candidates//2]])
            
            # Add some popular movies as backup
            popular_candidates = self.candidate_generator.generate_popularity_candidates(
                user_id=-1,
                num_candidates=num_candidates//2
            )
            candidates.extend(popular_candidates)
        
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
        # Create user feature vector
        user_features = self.create_user_features(user_demographics)
        user_features = user_features.unsqueeze(0).to(self.device)  # Add batch dimension
        
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
        # Get popular movies from different genres for diversity
        popular_movies = self.candidate_generator.generate_popularity_candidates(
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
        for genre, movies in genre_movies.items():
            if len(selected_movies) < num_movies and genre not in used_genres:
                selected_movies.append(movies[0])  # Take the most popular from this genre
                used_genres.add(genre)
        
        # Fill remaining slots with most popular movies
        for movie_id in popular_movies:
            if len(selected_movies) >= num_movies:
                break
            
            movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].iloc[0]
            movie_genres_str = self.movies_df[self.movies_df['movie_id'] == movie_id]['genres'].iloc[0]
            
            movie_tuple = (movie_id, movie_title, movie_genres_str)
            if movie_tuple not in selected_movies:
                selected_movies.append(movie_tuple)
        
        return selected_movies[:num_movies]
