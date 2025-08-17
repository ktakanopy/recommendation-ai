# Updated validation function for features-based models

import torch
from tqdm import tqdm

def validate_model_with_features(model, test_ratings, precomputed_candidates, device, total_users_to_test=20, k=10):
    """
    Validation function that works with features instead of IDs
    
    Args:
        model: The trained model (with feature processor)
        test_ratings: DataFrame with validation ratings
        precomputed_candidates: Dict of {user_id: [candidate_items]}
        device: torch device
        total_users_to_test: Number of users to test
        k: Top-k for hit ratio calculation
    """
    test_user_item_set = list(set(zip(test_ratings['user_id'], test_ratings['movie_id'])))
    hits = []
    ranks = []
    skipped_cases = 0
    total_cases = 0

    for (u, i) in tqdm(test_user_item_set[:total_users_to_test]):
        total_cases += 1
        
        # Check if user and movie are in our feature caches
        if u not in model.feature_processor.user_features_cache:
            print(f"Skipping user {u} - not in feature cache")
            skipped_cases += 1
            continue
            
        if i not in model.feature_processor.movie_features_cache:
            print(f"Skipping movie {i} - not in feature cache")
            skipped_cases += 1
            continue
        
        # Use precomputed candidates
        candidate_items = precomputed_candidates.get(u, [])
        
        # Make sure the test item is included
        if i not in candidate_items:
            candidate_items = candidate_items + [i]
        
        # Filter candidates to only include movies with features
        valid_candidates = [
            movie_id for movie_id in candidate_items 
            if movie_id in model.feature_processor.movie_features_cache
        ]
        
        if len(valid_candidates) == 0:
            print(f"No valid candidates for user {u}")
            skipped_cases += 1
            continue
            
        if i not in valid_candidates:
            print(f"Target movie {i} not in valid candidates for user {u}")
            skipped_cases += 1
            continue
        
        # Get user features
        user_feat = model.feature_processor.get_user_features(u).unsqueeze(0).to(device)
        
        # Score candidates using features
        predicted_scores = []
        for movie_id in valid_candidates:
            movie_feat = model.feature_processor.get_movie_features(movie_id).unsqueeze(0).to(device)
            with torch.no_grad():
                score = model(user_feat, movie_feat).item()
            predicted_scores.append(score)
        
        # Find rank and hit
        sorted_indices = sorted(range(len(predicted_scores)), key=lambda idx: predicted_scores[idx], reverse=True)
        sorted_items = [valid_candidates[idx] for idx in sorted_indices]
        relevant_item_rank = sorted_items.index(i) + 1
        ranks.append(relevant_item_rank)
        
        if relevant_item_rank <= k:
            hits.append(1)
        else:
            hits.append(0)
            
        # Debug: Print first few cases
        if len(hits) <= 3:
            target_score = predicted_scores[valid_candidates.index(i)] if i in valid_candidates else "N/A"
            print(f"User {u}, Movie {i}: Score={target_score}, Rank={relevant_item_rank}/{len(valid_candidates)}")

    print(f"\nValidation Summary:")
    print(f"Total test cases: {total_cases}")
    print(f"Skipped cases: {skipped_cases}")
    print(f"Valid cases processed: {len(hits)}")
    
    if len(hits) == 0:
        print("Warning: No valid test cases found!")
        return 0.0, 0.0, 0.0

    # Calculate metrics
    hits_tensor = torch.tensor(hits, dtype=torch.float32)
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    
    hit_ratio = hits_tensor.mean().item()
    mrr = torch.mean(1.0 / ranks_tensor).item()
    mean_rank = ranks_tensor.mean().item()
    
    print(f"Hit Ratio @ {k}: {hit_ratio:.3f}")
    print(f"Mean Rank: {mean_rank:.1f}")
    print(f"MRR: {mrr:.3f}")
    
    return hit_ratio, mrr, mean_rank
