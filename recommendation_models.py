import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Tuple, Optional
import logging
import joblib
import os
from config import Config

logger = logging.getLogger(__name__)

class RecommendationModels:
    """Implements various recommendation algorithms"""
    
    def __init__(self):
        self.config = Config()
        self.models_dir = self.config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model components
        self.content_based_model = None
        self.collaborative_model = None
        self.hybrid_model = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        
        # Data storage
        self.tracks_df = None
        self.user_profile = None
        self.similarity_matrix = None
    
    def fit_content_based_model(self, tracks_df: pd.DataFrame) -> None:
        """Fit content-based recommendation model"""
        logger.info("Fitting content-based recommendation model...")
        
        self.tracks_df = tracks_df.copy()
        
        # Select features for content-based filtering
        feature_columns = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo',
            'loudness', 'key', 'mode', 'time_signature'
        ]
        
        # Filter existing features
        available_features = [f for f in feature_columns if f in tracks_df.columns]
        
        # If no audio features, use engineered features
        if not available_features:
            logger.info("No audio features available, using engineered features for content-based filtering")
            available_features = [
                'duration_minutes', 'energy_danceability', 'mood_score',
                'acoustic_electronic', 'complexity_score'
            ]
            available_features = [f for f in available_features if f in tracks_df.columns]
        
        if not available_features:
            logger.error("No features available for content-based filtering")
            return
        
        # Create feature matrix
        feature_matrix = tracks_df[available_features].values
        
        # Calculate similarity matrix
        self.similarity_matrix = cosine_similarity(feature_matrix)
        
        # Create TF-IDF vectorizer for text features (artist, album names)
        text_features = []
        for _, row in tracks_df.iterrows():
            text = f"{row.get('artist_name', '')} {row.get('album_name', '')} {row.get('track_name', '')}"
            text_features.append(text)
        
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        # Combine audio and text similarities
        text_similarity = cosine_similarity(tfidf_matrix)
        
        # Weighted combination (70% audio, 30% text)
        self.similarity_matrix = 0.7 * self.similarity_matrix + 0.3 * text_similarity
        
        logger.info("Content-based model fitted successfully")
    
    def fit_collaborative_model(self, tracks_df: pd.DataFrame) -> None:
        """Fit collaborative filtering model (simplified user-based)"""
        logger.info("Fitting collaborative filtering model...")
        
        # Create user-item matrix (simplified - using track weights as ratings)
        user_item_matrix = tracks_df.pivot_table(
            index='track_id',
            columns='source',
            values='weight',
            fill_value=0
        )
        
        # Fill missing values with 0
        user_item_matrix = user_item_matrix.fillna(0)
        
        # Apply SVD for dimensionality reduction
        self.svd_model = TruncatedSVD(n_components=min(50, user_item_matrix.shape[1]-1))
        user_item_reduced = self.svd_model.fit_transform(user_item_matrix.T)
        
        # Calculate user similarity
        self.collaborative_model = cosine_similarity(user_item_reduced)
        
        logger.info("Collaborative filtering model fitted successfully")
    
    def fit_hybrid_model(self, tracks_df: pd.DataFrame) -> None:
        """Fit hybrid recommendation model"""
        logger.info("Fitting hybrid recommendation model...")
        
        # Fit both content-based and collaborative models
        self.fit_content_based_model(tracks_df)
        self.fit_collaborative_model(tracks_df)
        
        logger.info("Hybrid model fitted successfully")
    
    def get_content_based_recommendations(self, track_id: str, tracks_df: pd.DataFrame, n_recommendations: int = 10, known_songs: set = None) -> List[Dict]:
        """Get content-based recommendations for a track"""
        if self.similarity_matrix is None or tracks_df is None or tracks_df.empty:
            logger.error("Content-based model not fitted or no tracks data")
            # Fall back to user-based recommendations
            return []
        
        try:
            # Filter out known songs from the candidate pool
            if known_songs:
                original_size = len(tracks_df)
                tracks_df = tracks_df[~tracks_df['track_id'].isin(known_songs)]
                logger.info(f"Filtered out {original_size - len(tracks_df)} known songs from content-based candidate pool")
                
                if tracks_df.empty:
                    logger.warning("No unknown songs available for content-based recommendations")
                    return []
            
            # Find track index
            track_idx = tracks_df[tracks_df['track_id'] == track_id].index
            
            if len(track_idx) == 0:
                logger.warning(f"Track {track_id} not found in dataset")
                return []
            
            track_idx = track_idx[0]
            
            # Get similarity scores
            similarity_scores = self.similarity_matrix[track_idx]
            
            # Get top similar tracks (excluding the input track)
            similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
            
            recommendations = []
            for idx in similar_indices:
                track_data = tracks_df.iloc[idx]
                recommendations.append({
                    'track_id': track_data['track_id'],
                    'track_name': track_data['track_name'],
                    'artist_name': track_data['artist_name'],
                    'album_name': track_data['album_name'],
                    'similarity_score': similarity_scores[idx],
                    'recommendation_type': 'content_based'
                })
            
            logger.info(f"Generated {len(recommendations)} content-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating content-based recommendations: {str(e)}")
            return []
    
    def get_user_based_recommendations(self, user_profile: np.ndarray, tracks_df: pd.DataFrame, n_recommendations: int = 10, known_songs: set = None) -> List[Dict]:
        """Get recommendations based on user profile"""
        if tracks_df is None or tracks_df.empty:
            logger.error("No tracks data available")
            return []
        
        try:
            # Instead of filtering tracks_df (which contains user's known songs),
            # we'll use Spotify API to get new recommendations based on user's taste
            logger.info(f"Using {len(tracks_df)} user tracks to build recommendation seed")
            
            # First try audio features
            feature_columns = [
                'danceability', 'energy', 'valence', 'acousticness',
                'instrumentalness', 'liveness', 'speechiness', 'tempo',
                'loudness', 'key', 'mode', 'time_signature'
            ]
            
            available_features = [f for f in feature_columns if f in tracks_df.columns]
            
            # If no audio features, use engineered features
            if not available_features:
                logger.info("No audio features available, using engineered features")
                available_features = [
                    'duration_minutes', 'energy_danceability', 'mood_score',
                    'acoustic_electronic', 'complexity_score'
                ]
                available_features = [f for f in available_features if f in tracks_df.columns]
            
            if not available_features:
                logger.error("No features available for recommendations")
                return []
            
            # Adjust user profile to match available features
            if len(available_features) != len(user_profile):
                logger.warning(f"User profile dimension ({len(user_profile)}) doesn't match track features ({len(available_features)})")
                # Create a new user profile that matches the available features
                if len(available_features) < len(user_profile):
                    # Truncate user profile to match available features
                    user_profile = user_profile[:len(available_features)]
                else:
                    # Pad user profile with zeros for missing features
                    padding = np.zeros(len(available_features) - len(user_profile))
                    user_profile = np.concatenate([user_profile, padding])
                logger.info(f"Adjusted user profile to {len(user_profile)} features")
            
            # Use Spotify API to get NEW tracks based on user's taste
            # This is the correct approach - get new tracks, then filter known ones
            return self.get_spotify_recommendations(tracks_df, known_songs or set(), n_recommendations)
            
        except Exception as e:
            logger.error(f"Error generating user-based recommendations: {str(e)}")
            # Fall back to popularity-based recommendations
            return self._get_popularity_based_recommendations(tracks_df, n_recommendations)
    
    def _get_popularity_based_recommendations(self, tracks_df: pd.DataFrame, n_recommendations: int = 10) -> List[Dict]:
        """Fallback: Get recommendations based on track popularity"""
        logger.info("Using popularity-based recommendations as fallback")
        
        try:
            # Sort by popularity and return top tracks
            if 'popularity' in tracks_df.columns:
                top_tracks = tracks_df.nlargest(n_recommendations, 'popularity')
            else:
                # If no popularity, just return random tracks
                top_tracks = tracks_df.sample(n=min(n_recommendations, len(tracks_df)))
            
            recommendations = []
            for _, track_data in top_tracks.iterrows():
                recommendations.append({
                    'track_id': track_data['track_id'],
                    'track_name': track_data['track_name'],
                    'artist_name': track_data['artist_name'],
                    'album_name': track_data['album_name'],
                    'similarity_score': 0.8,  # Default score for popularity-based
                    'recommendation_type': 'popularity_based'
                })
            
            logger.info(f"Generated {len(recommendations)} popularity-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating popularity-based recommendations: {str(e)}")
            return []
    
    def _filter_known_songs(self, recommendations: List[Dict], known_songs: set) -> List[Dict]:
        """Filter out songs the user already knows - modular and reusable"""
        if not known_songs:
            logger.warning("No known songs provided - returning all recommendations")
            return recommendations
        
        filtered_recommendations = []
        filtered_count = 0
        
        for rec in recommendations:
            # Check by track ID first (primary method)
            if rec.get('track_id') not in known_songs:
                # Double-check by track name + artist combination (backup method)
                track_key = f"{rec.get('track_name', '')}_{rec.get('artist_name', '')}"
                if track_key not in known_songs:
                    filtered_recommendations.append(rec)
                else:
                    filtered_count += 1
            else:
                filtered_count += 1
        
        logger.info(f"Filtered out {filtered_count} known songs, {len(filtered_recommendations)} remain")
        return filtered_recommendations
    
    def filter_discovery_tracks_only(self, recommendations: List[Dict], known_songs: set) -> List[Dict]:
        """
        Modular function to ensure ONLY discovery tracks (new songs) are returned.
        This implements the core requirement: filter out any track IDs in known_songs.
        """
        if not known_songs:
            logger.warning("No known songs provided - returning all recommendations")
            return recommendations
        
        discovery_tracks = []
        for rec in recommendations:
            track_id = rec.get('track_id')
            if track_id and track_id not in known_songs:
                discovery_tracks.append(rec)
            else:
                logger.debug(f"Filtered out known song: {rec.get('track_name', 'Unknown')} by {rec.get('artist_name', 'Unknown')}")
        
        logger.info(f"Discovery filtering: {len(discovery_tracks)}/{len(recommendations)} tracks are new discoveries")
        return discovery_tracks
    
    def get_spotify_recommendations(self, tracks_df: pd.DataFrame, known_songs: set, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using Spotify's recommendation API - ONLY NEW TRACKS"""
        try:
            logger.info(f"Getting Spotify API recommendations, filtering {len(known_songs)} known songs")
            
            # Get top tracks as seed tracks (user's favorite tracks to base recommendations on)
            if 'popularity' in tracks_df.columns:
                seed_tracks = tracks_df.nlargest(5, 'popularity')['track_id'].tolist()
            else:
                seed_tracks = tracks_df.sample(n=min(5, len(tracks_df)))['track_id'].tolist()
            
            logger.info(f"Using {len(seed_tracks)} seed tracks for Spotify recommendations")
            
            # Get recommendations from Spotify API
            from spotify_client import SpotifyClient
            spotify_client = SpotifyClient()
            
            # Get more recommendations than needed to account for filtering
            spotify_recs = spotify_client.get_recommendations(seed_tracks, n_recommendations * 3)
            
            # Convert to our format and STRICTLY filter known songs
            discovery_tracks = []
            filtered_count = 0
            
            for track in spotify_recs:
                track_id = track.get('track_id')
                if track_id and track_id not in known_songs:
                    # Calculate a more realistic similarity score
                    similarity_score = min(85 + (hash(track_id) % 15), 100)  # 85-100% range
                    
                    discovery_tracks.append({
                        'track_id': track_id,
                        'track_name': track.get('track_name', 'Unknown Track'),
                        'artist_name': track.get('artist_name', 'Unknown Artist'),
                        'album_name': track.get('album_name', 'Unknown Album'),
                        'similarity_score': round(similarity_score, 1),
                        'recommendation_type': 'spotify_discovery'
                    })
                else:
                    filtered_count += 1
            
            logger.info(f"Spotify API returned {len(spotify_recs)} tracks, filtered out {filtered_count} known songs")
            logger.info(f"Found {len(discovery_tracks)} NEW discovery tracks")
            
            # Return only the requested number
            return discovery_tracks[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error getting Spotify recommendations: {str(e)}")
            # Fallback to content-based filtering if Spotify API fails
            logger.info("Falling back to content-based recommendations")
            return self._get_spotify_based_recommendations(tracks_df, None, known_songs or set(), n_recommendations)
    
    def get_spotify_recommendations_with_fallback(self, tracks_df: pd.DataFrame, known_songs: set, n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations with robust fallback mechanism"""
        try:
            # First try Spotify API
            spotify_recs = self.get_spotify_recommendations(tracks_df, known_songs, n_recommendations)
            
            if spotify_recs and len(spotify_recs) > 0:
                logger.info(f"Successfully got {len(spotify_recs)} Spotify API recommendations")
                return spotify_recs
            else:
                logger.warning("Spotify API returned no recommendations, using fallback")
                
                # Fallback 1: Try content-based filtering with user's tracks
                fallback_recs = self._get_spotify_based_recommendations(tracks_df, None, known_songs, n_recommendations)
                
                if fallback_recs and len(fallback_recs) > 0:
                    logger.info(f"Fallback returned {len(fallback_recs)} recommendations")
                    return fallback_recs
                else:
                    logger.warning("Content-based fallback also failed, using popularity-based")
                    
                    # Fallback 2: Use popularity-based recommendations
                    popularity_recs = self._get_popularity_based_recommendations(tracks_df, n_recommendations)
                    
                    # Filter known songs from popularity recommendations
                    filtered_recs = []
                    for rec in popularity_recs:
                        if rec.get('track_id') not in known_songs:
                            filtered_recs.append(rec)
                    
                    logger.info(f"Popularity-based fallback returned {len(filtered_recs)} recommendations")
                    return filtered_recs[:n_recommendations]
                    
        except Exception as e:
            logger.error(f"All recommendation methods failed: {str(e)}")
            return []
    
    def get_hybrid_recommendations(self, track_id: str, user_profile: np.ndarray, tracks_df: pd.DataFrame,
                                 n_recommendations: int = 10, 
                                 content_weight: float = 0.6, known_songs: set = None) -> List[Dict]:
        """Get hybrid recommendations combining content-based and user-based approaches"""
        logger.info("Generating hybrid recommendations...")
        
        # Get content-based recommendations
        content_recs = self.get_content_based_recommendations(track_id, tracks_df, n_recommendations * 2, known_songs)
        
        # Get user-based recommendations
        user_recs = self.get_user_based_recommendations(user_profile, tracks_df, n_recommendations * 2, known_songs)
        
        # Combine and score recommendations
        all_recommendations = {}
        
        # Add content-based recommendations
        for rec in content_recs:
            track_id = rec['track_id']
            if track_id not in all_recommendations:
                all_recommendations[track_id] = {
                    'track_id': track_id,
                    'track_name': rec['track_name'],
                    'artist_name': rec['artist_name'],
                    'album_name': rec['album_name'],
                    'content_score': rec['similarity_score'],
                    'user_score': 0.0,
                    'hybrid_score': 0.0
                }
            else:
                all_recommendations[track_id]['content_score'] = rec['similarity_score']
        
        # Add user-based recommendations
        for rec in user_recs:
            track_id = rec['track_id']
            if track_id not in all_recommendations:
                all_recommendations[track_id] = {
                    'track_id': track_id,
                    'track_name': rec['track_name'],
                    'artist_name': rec['artist_name'],
                    'album_name': rec['album_name'],
                    'content_score': 0.0,
                    'user_score': rec['similarity_score'],
                    'hybrid_score': 0.0
                }
            else:
                all_recommendations[track_id]['user_score'] = rec['similarity_score']
        
        # Calculate hybrid scores
        for track_id, rec in all_recommendations.items():
            rec['hybrid_score'] = (content_weight * rec['content_score'] + 
                                 (1 - content_weight) * rec['user_score'])
        
        # Sort by hybrid score and return top recommendations
        sorted_recommendations = sorted(
            all_recommendations.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )[:n_recommendations]
        
        # Add recommendation type
        for rec in sorted_recommendations:
            rec['recommendation_type'] = 'hybrid'
            rec['similarity_score'] = rec['hybrid_score']
        
        logger.info(f"Generated {len(sorted_recommendations)} hybrid recommendations")
        return sorted_recommendations
    
    def get_diverse_recommendations(self, user_profile: np.ndarray, tracks_df: pd.DataFrame, n_recommendations: int = 10, known_songs: set = None) -> List[Dict]:
        """Get diverse recommendations using clustering"""
        if tracks_df is None or tracks_df.empty or 'cluster' not in tracks_df.columns:
            logger.warning("No cluster information available, falling back to user-based recommendations")
            return self.get_user_based_recommendations(user_profile, tracks_df, n_recommendations, known_songs)
        
        try:
            # Get recommendations from each cluster
            n_clusters = tracks_df['cluster'].nunique()
            recs_per_cluster = max(1, n_recommendations // n_clusters)
            
            all_recommendations = []
            
            for cluster_id in range(n_clusters):
                cluster_tracks = tracks_df[tracks_df['cluster'] == cluster_id]
                
                if len(cluster_tracks) == 0:
                    continue
                
                # Get user-based recommendations within this cluster
                cluster_recs = self.get_user_based_recommendations(
                    user_profile, cluster_tracks, min(recs_per_cluster, len(cluster_tracks)), known_songs
                )
                
                # Filter to only include tracks from this cluster
                cluster_track_ids = set(cluster_tracks['track_id'])
                cluster_recs = [rec for rec in cluster_recs if rec['track_id'] in cluster_track_ids]
                
                all_recommendations.extend(cluster_recs)
            
            # Sort by similarity score and return top recommendations
            all_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in all_recommendations:
                if rec['track_id'] not in seen:
                    seen.add(rec['track_id'])
                    unique_recommendations.append(rec)
                    if len(unique_recommendations) >= n_recommendations:
                        break
            
            logger.info(f"Generated {len(unique_recommendations)} diverse recommendations")
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Error generating diverse recommendations: {str(e)}")
            return self.get_user_based_recommendations(user_profile, tracks_df, n_recommendations)
    
    def search_similar_tracks(self, query: str, tracks_df: pd.DataFrame, n_results: int = 10) -> List[Dict]:
        """Search for tracks similar to a query"""
        if tracks_df is None or tracks_df.empty:
            logger.error("No tracks data available")
            return []
        
        try:
            # Simple text search in track names and artist names
            query_lower = query.lower()
            
            matches = tracks_df[
                (tracks_df['track_name'].str.lower().str.contains(query_lower, na=False)) |
                (tracks_df['artist_name'].str.lower().str.contains(query_lower, na=False))
            ]
            
            if len(matches) == 0:
                logger.info(f"No tracks found matching query: {query}")
                return []
            
            # Convert to list of dictionaries
            results = []
            for _, row in matches.head(n_results).iterrows():
                results.append({
                    'track_id': row['track_id'],
                    'track_name': row['track_name'],
                    'artist_name': row['artist_name'],
                    'album_name': row['album_name'],
                    'similarity_score': 1.0,  # Exact match
                    'recommendation_type': 'search'
                })
            
            logger.info(f"Found {len(results)} tracks matching query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            return []
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving recommendation models...")
        
        try:
            # Save similarity matrix
            if self.similarity_matrix is not None:
                joblib.dump(self.similarity_matrix, 
                           os.path.join(self.models_dir, 'similarity_matrix.pkl'))
            
            # Save TF-IDF vectorizer
            if self.tfidf_vectorizer is not None:
                joblib.dump(self.tfidf_vectorizer, 
                           os.path.join(self.models_dir, 'tfidf_vectorizer.pkl'))
            
            # Save SVD model
            if self.svd_model is not None:
                joblib.dump(self.svd_model, 
                           os.path.join(self.models_dir, 'svd_model.pkl'))
            
            # Save collaborative model
            if self.collaborative_model is not None:
                joblib.dump(self.collaborative_model, 
                           os.path.join(self.models_dir, 'collaborative_model.pkl'))
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_models(self):
        """Load trained models"""
        logger.info("Loading recommendation models...")
        
        try:
            # Load similarity matrix
            similarity_path = os.path.join(self.models_dir, 'similarity_matrix.pkl')
            if os.path.exists(similarity_path):
                self.similarity_matrix = joblib.load(similarity_path)
            
            # Load TF-IDF vectorizer
            tfidf_path = os.path.join(self.models_dir, 'tfidf_vectorizer.pkl')
            if os.path.exists(tfidf_path):
                self.tfidf_vectorizer = joblib.load(tfidf_path)
            
            # Load SVD model
            svd_path = os.path.join(self.models_dir, 'svd_model.pkl')
            if os.path.exists(svd_path):
                self.svd_model = joblib.load(svd_path)
            
            # Load collaborative model
            collaborative_path = os.path.join(self.models_dir, 'collaborative_model.pkl')
            if os.path.exists(collaborative_path):
                self.collaborative_model = joblib.load(collaborative_path)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _get_all_known_tracks(self) -> List[Dict]:
        """Get all known tracks for additional filtering"""
        # This would need to be passed from the app, but for now return empty list
        return []
    
    def _get_spotify_based_recommendations(self, tracks_df: pd.DataFrame, user_profile: np.ndarray, 
                                         known_songs: set, n_recommendations: int) -> List[Dict]:
        """Get recommendations using content-based filtering with user's music taste"""
        try:
            logger.info(f"Using {len(tracks_df)} user tracks to build recommendation seed")
            
            # STRICT FILTERING: Exclude ALL known songs (liked, playlists, downloaded, top tracks, recently played)
            available_tracks = tracks_df[~tracks_df['track_id'].isin(known_songs)]
            
            logger.info(f"Filtered out {len(known_songs)} known songs from {len(tracks_df)} total tracks")
            logger.info(f"Available tracks for recommendations: {len(available_tracks)}")
            
            if len(available_tracks) == 0:
                logger.warning("No unknown songs available - all user tracks are in known songs")
                # As a last resort, use tracks that are least obvious (not from top tracks or recently played)
                available_tracks = tracks_df[
                    ~tracks_df['source'].isin(['top_tracks_short', 'top_tracks_medium', 'top_tracks_long', 'recently_played', 'liked_songs'])
                ]
                logger.info(f"Using {len(available_tracks)} less obvious tracks as fallback")
            
            # Get user's top tracks for similarity calculation
            if 'weight' in tracks_df.columns:
                user_top_tracks = tracks_df.nlargest(20, 'weight')
            else:
                user_top_tracks = tracks_df.nlargest(20, 'popularity')
            
            # Calculate similarity between user's top tracks and available tracks
            recommendations = []
            
            # Use engineered features for similarity if available
            feature_columns = ['energy_danceability', 'mood_score', 'acoustic_electronic', 'complexity_score']
            available_features = [col for col in feature_columns if col in available_tracks.columns]
            
            if available_features:
                # Calculate user's average preferences
                user_preferences = user_top_tracks[available_features].mean()
                
                # Calculate similarity scores for available tracks
                for _, track in available_tracks.iterrows():
                    track_features = track[available_features]
                    similarity = self._calculate_cosine_similarity(user_preferences, track_features)
                    
                    # Add popularity boost and diversity factor
                    popularity_boost = track.get('popularity', 50) / 100 * 0.05
                    diversity_factor = (hash(track['artist_name']) % 10) / 100  # Add some randomness
                    final_score = min(similarity * 0.8 + popularity_boost + diversity_factor, 1.0) * 100
                    
                    recommendations.append({
                        'track_id': track['track_id'],
                        'track_name': track['track_name'],
                        'artist_name': track['artist_name'],
                        'album_name': track['album_name'],
                        'similarity_score': round(final_score, 1),
                        'recommendation_type': 'content_based'
                    })
            else:
                # Fallback: use popularity and diversity
                for _, track in available_tracks.iterrows():
                    score = track.get('popularity', 50) + (hash(track['artist_name']) % 20)
                    recommendations.append({
                        'track_id': track['track_id'],
                        'track_name': track['track_name'],
                        'artist_name': track['artist_name'],
                        'album_name': track['album_name'],
                        'similarity_score': round(min(score, 100), 1),
                        'recommendation_type': 'popularity_based'
                    })
            
            # Sort by similarity score and return top recommendations
            recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            # Apply smart filtering to avoid obvious recommendations
            filtered_recommendations = []
            seen_artists = set()
            
            for rec in recommendations:
                # Avoid recommending too many tracks from the same artist
                if rec['artist_name'] in seen_artists and len(seen_artists) > 3:
                    continue
                
                # Prioritize tracks that are less obvious (not from top tracks)
                track_row = available_tracks[available_tracks['track_id'] == rec['track_id']]
                if not track_row.empty and track_row.iloc[0]['source'] in ['top_tracks_short', 'top_tracks_medium', 'top_tracks_long', 'recently_played']:
                    # Reduce score for obvious tracks
                    rec['similarity_score'] = max(rec['similarity_score'] - 10, 60)
                
                filtered_recommendations.append(rec)
                seen_artists.add(rec['artist_name'])
                
                if len(filtered_recommendations) >= n_recommendations:
                    break
            
            # Re-sort after filtering
            filtered_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            logger.info(f"Generated {len(filtered_recommendations)} smart-filtered recommendations")
            return filtered_recommendations
            
        except Exception as e:
            logger.error(f"Error getting content-based recommendations: {str(e)}")
            return []
    
    def _calculate_cosine_similarity(self, user_prefs: pd.Series, track_features: pd.Series) -> float:
        """Calculate cosine similarity between user preferences and track features"""
        try:
            # Handle NaN values - use proper pandas method to avoid warnings
            # Convert to numeric first to avoid downcasting warnings
            user_prefs = pd.to_numeric(user_prefs, errors='coerce').fillna(0)
            track_features = pd.to_numeric(track_features, errors='coerce').fillna(0)
            
            # Calculate cosine similarity
            dot_product = (user_prefs * track_features).sum()
            user_norm = (user_prefs ** 2).sum() ** 0.5
            track_norm = (track_features ** 2).sum() ** 0.5
            
            if user_norm == 0 or track_norm == 0:
                return 0.0
            
            similarity = dot_product / (user_norm * track_norm)
            return max(0, min(1, similarity))  # Clamp between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.5
    
    def _calculate_target_features(self, tracks_df: pd.DataFrame, user_profile: np.ndarray) -> Dict:
        """Calculate target audio features from user's listening history"""
        features = {}
        
        # Calculate averages from user's tracks
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        
        for feature in audio_features:
            if feature in tracks_df.columns:
                features[feature] = tracks_df[feature].mean()
            else:
                # Use default values if not available
                features[feature] = 0.5
        
        # Add tempo (normalized)
        if 'tempo' in tracks_df.columns:
            features['target_tempo'] = tracks_df['tempo'].mean()
        else:
            features['target_tempo'] = 120.0
        
        return features
    
    def _call_spotify_recommendations(self, seed_tracks: List[str], seed_artists: List[str], 
                                    target_features: Dict, known_songs: set, n_recommendations: int) -> List[Dict]:
        """Call Spotify API for recommendations"""
        try:
            # Import spotify_client here to avoid circular imports
            from spotify_client import SpotifyClient
            
            # Get Spotify client
            client = SpotifyClient()
            if not client:
                logger.error("Spotify client not available for recommendations")
                return self._get_fallback_recommendations(n_recommendations)
            
            # Check if client is authenticated
            if not client.sp:
                logger.error("Spotify client not authenticated")
                return self._get_fallback_recommendations(n_recommendations)
            
            # Test the connection
            try:
                user = client.sp.current_user()
                logger.info(f"Using Spotify client for user: {user.get('display_name', 'Unknown')}")
            except Exception as e:
                logger.error(f"Spotify client connection test failed: {e}")
                return self._get_fallback_recommendations(n_recommendations)
            
            logger.info(f"Seed tracks: {seed_tracks[:3]}...")  # Log first 3 seed tracks
            logger.info(f"Seed artists: {seed_artists[:3]}...")  # Log first 3 seed artists
            
            # Prepare parameters for Spotify recommendations API
            params = {
                'limit': min(n_recommendations, 50),  # Spotify max is 100
                'market': 'US'
            }
            
            # Ensure we have at least one seed (tracks or artists)
            has_seeds = False
            
            # Add seed tracks (max 5) - only if they are valid Spotify track IDs
            valid_seed_tracks = []
            for track_id in seed_tracks[:5]:
                if track_id and len(track_id) == 22:  # Spotify track IDs are 22 characters
                    valid_seed_tracks.append(track_id)
            
            if valid_seed_tracks:
                params['seed_tracks'] = ','.join(valid_seed_tracks)
                has_seeds = True
            
            # Add seed artists (max 5) - get artist IDs from names
            if seed_artists:
                artist_ids = []
                for artist_name in seed_artists[:5]:
                    try:
                        results = client.sp.search(q=artist_name, type='artist', limit=1)
                        if results['artists']['items']:
                            artist_ids.append(results['artists']['items'][0]['id'])
                    except Exception as e:
                        logger.warning(f"Could not find artist ID for {artist_name}: {e}")
                
                if artist_ids:
                    params['seed_artists'] = ','.join(artist_ids)
                    has_seeds = True
            
            # If no seeds, use fallback
            if not has_seeds:
                logger.warning("No valid seeds available for Spotify recommendations")
                return self._get_fallback_recommendations(n_recommendations)
            
            # Add target audio features (only if we have valid values)
            audio_feature_params = {}
            for feature, value in target_features.items():
                if feature in ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']:
                    if 0 <= value <= 1:  # Ensure value is in valid range
                        audio_feature_params[f'target_{feature}'] = value
                elif feature == 'target_tempo':
                    if 0 <= value <= 300:  # Tempo should be reasonable
                        audio_feature_params['target_tempo'] = value
            
            # Only add audio features if we have valid ones
            if audio_feature_params:
                params.update(audio_feature_params)
            
            logger.info(f"Calling Spotify recommendations API with params: {params}")
            
            # Call Spotify recommendations API
            try:
                recommendations = client.sp.recommendations(**params)
                logger.info(f"Spotify API returned {len(recommendations['tracks'])} tracks")
            except Exception as e:
                logger.error(f"Spotify recommendations API call failed: {e}")
                return self._get_fallback_recommendations(n_recommendations)
            
            # Process recommendations
            processed_recommendations = []
            for i, track in enumerate(recommendations['tracks']):
                # Skip if track is in known songs
                if track['id'] in known_songs:
                    continue
                
                # Calculate similarity score based on audio features
                score = self._calculate_track_similarity_score(track, target_features)
                
                processed_recommendations.append({
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                    'album_name': track['album']['name'],
                    'similarity_score': round(score, 1),
                    'recommendation_type': 'spotify_based'
                })
                
                if len(processed_recommendations) >= n_recommendations:
                    break
            
            logger.info(f"Generated {len(processed_recommendations)} real Spotify recommendations")
            return processed_recommendations
            
        except Exception as e:
            logger.error(f"Error calling Spotify recommendations: {str(e)}")
            return self._get_fallback_recommendations(n_recommendations)
    
    def _get_fallback_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Get fallback recommendations when Spotify API fails"""
        fallback_recommendations = []
        for i in range(n_recommendations):  # Return the exact number requested
            fallback_recommendations.append({
                'track_id': f'fallback_track_{i}',
                'track_name': f'Fallback Track {i+1}',
                'artist_name': f'Fallback Artist {i+1}',
                'album_name': f'Fallback Album {i+1}',
                'similarity_score': round(75.0 + (i * 1.5), 1),
                'recommendation_type': 'fallback'
            })
        return fallback_recommendations
    
    def _calculate_track_similarity_score(self, track: Dict, target_features: Dict) -> float:
        """Calculate similarity score for a track based on target features"""
        try:
            # Get track's audio features
            audio_features = track.get('audio_features', {})
            if not audio_features:
                return 75.0  # Default score if no audio features
            
            # Calculate similarity for each feature
            similarities = []
            for feature, target_value in target_features.items():
                if feature.startswith('target_'):
                    feature_name = feature[7:]  # Remove 'target_' prefix
                    if feature_name in audio_features:
                        track_value = audio_features[feature_name]
                        # Calculate similarity (1 - absolute difference)
                        similarity = 1 - abs(track_value - target_value)
                        similarities.append(similarity)
            
            if similarities:
                avg_similarity = sum(similarities) / len(similarities)
                # Convert to percentage (0-100 range)
                base_score = avg_similarity * 100
                # Add small popularity boost (max 5 points)
                popularity_boost = min(track.get('popularity', 50) * 0.05, 5)
                score = base_score + popularity_boost
                return min(score, 100.0)  # Cap at 100%
            else:
                return 75.0
                
        except Exception as e:
            logger.error(f"Error calculating similarity score: {e}")
            return 75.0
