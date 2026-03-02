import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from spotify_client import SpotifyClient
from config import Config

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects and manages user data from Spotify"""
    
    def __init__(self, spotify_client=None):
        self.config = Config()
        self.spotify_client = spotify_client
        self.data_dir = self.config.DATA_DIR
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def collect_user_data(self, save_to_file: bool = True) -> Dict:
        """Collect comprehensive user data from Spotify"""
        logger.info("Starting data collection...")
        
        if self.spotify_client is None:
            raise ValueError("Spotify client not initialized. Please provide a valid SpotifyClient instance.")
        
        user_data = {
            'collection_timestamp': datetime.now().isoformat(),
            'recently_played': [],
            'top_tracks_short': [],
            'top_tracks_medium': [],
            'top_tracks_long': [],
            'audio_features': [],
            'playlists': []
        }
        
        try:
            # Collect recently played tracks
            logger.info("Collecting recently played tracks...")
            user_data['recently_played'] = self.spotify_client.get_recently_played(limit=50)
            
            # Collect top tracks for different time ranges
            time_ranges = ['short_term', 'medium_term', 'long_term']
            time_range_names = ['top_tracks_short', 'top_tracks_medium', 'top_tracks_long']
            
            for time_range, name in zip(time_ranges, time_range_names):
                logger.info(f"Collecting {name}...")
                user_data[name] = self.spotify_client.get_top_tracks(
                    time_range=time_range, limit=50
                )
            
            # Collect playlists (comprehensive)
            logger.info("Collecting playlists (comprehensive)...")
            user_data['playlists'] = self.spotify_client.get_user_playlists_comprehensive(limit=1000)
            
            # Collect liked songs (comprehensive)
            logger.info("Collecting liked songs (comprehensive)...")
            user_data['liked_songs'] = self.spotify_client.get_user_saved_tracks_comprehensive(limit=1000)
            
            # Collect saved albums (comprehensive)
            logger.info("Collecting saved albums (comprehensive)...")
            user_data['saved_albums'] = self.spotify_client.get_user_saved_albums_comprehensive(limit=1000)
            
            # Collect tracks from saved albums
            logger.info("Collecting tracks from saved albums...")
            user_data['saved_album_tracks'] = {}
            for album in user_data['saved_albums']:
                album_id = album['album_id']
                tracks = self.spotify_client.get_album_tracks(album_id)
                user_data['saved_album_tracks'][album_id] = tracks
            
            # Collect tracks from all playlists
            logger.info("Collecting tracks from playlists...")
            user_data['playlist_tracks'] = {}
            for playlist in user_data['playlists']:
                playlist_id = playlist['playlist_id']
                tracks = self.spotify_client.get_playlist_tracks(playlist_id)
                user_data['playlist_tracks'][playlist_id] = tracks
            
            # Get all unique track IDs from all sources
            all_track_ids = set()
            
            # Add tracks from basic categories
            for category in ['recently_played', 'top_tracks_short', 'top_tracks_medium', 'top_tracks_long']:
                for track in user_data[category]:
                    all_track_ids.add(track['track_id'])
            
            # Add liked songs
            for track in user_data['liked_songs']:
                all_track_ids.add(track['track_id'])
            
            # Add playlist tracks
            for playlist_id, tracks in user_data['playlist_tracks'].items():
                for track in tracks:
                    all_track_ids.add(track['track_id'])
            
            # Get audio features for all tracks
            if all_track_ids:
                logger.info(f"Collecting audio features for {len(all_track_ids)} tracks...")
                user_data['audio_features'] = self.spotify_client.get_audio_features(list(all_track_ids))
            
            # Save to file if requested
            if save_to_file:
                self._save_user_data(user_data)
            
            logger.info("Data collection completed successfully")
            return user_data
            
        except Exception as e:
            logger.error(f"Error during data collection: {str(e)}")
            raise
    
    def _save_user_data(self, user_data: Dict):
        """Save user data to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"user_data_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"User data saved to {filepath}")
    
    def load_latest_user_data(self) -> Optional[Dict]:
        """Load the most recent user data file"""
        try:
            data_files = [f for f in os.listdir(self.data_dir) if f.startswith('user_data_') and f.endswith('.json')]
            
            if not data_files:
                logger.warning("No user data files found")
                return None
            
            # Sort by timestamp and get the latest
            latest_file = sorted(data_files)[-1]
            filepath = os.path.join(self.data_dir, latest_file)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
            
            logger.info(f"Loaded user data from {latest_file}")
            return user_data
            
        except Exception as e:
            logger.error(f"Error loading user data: {str(e)}")
            return None
    
    def create_tracks_dataframe(self, user_data: Dict) -> pd.DataFrame:
        """Create a comprehensive tracks DataFrame from user data"""
        tracks_data = []
        
        # Process each category of tracks with different weights
        categories = {
            'recently_played': {'weight': 1.0, 'source': 'recent'},
            'top_tracks_short': {'weight': 2.0, 'source': 'top_short'},
            'top_tracks_medium': {'weight': 3.0, 'source': 'top_medium'},
            'top_tracks_long': {'weight': 4.0, 'source': 'top_long'},
            'liked_songs': {'weight': 5.0, 'source': 'liked'}  # Liked songs have highest weight
        }
        
        for category, config in categories.items():
            for track in user_data.get(category, []):
                track_data = {
                    'track_id': track['track_id'],
                    'track_name': track['track_name'],
                    'artist_name': track['artist_name'],
                    'album_name': track['album_name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'explicit': track['explicit'],
                    'source': config['source'],
                    'weight': config['weight'],
                    'played_at': track.get('played_at', None)
                }
                tracks_data.append(track_data)
        
        # Process playlist tracks (with lower weight since they're in playlists)
        for playlist_id, tracks in user_data.get('playlist_tracks', {}).items():
            for track in tracks:
                track_data = {
                    'track_id': track['track_id'],
                    'track_name': track['track_name'],
                    'artist_name': track['artist_name'],
                    'album_name': track['album_name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'explicit': track['explicit'],
                    'source': 'playlist',
                    'weight': 1.5,  # Lower weight for playlist tracks
                    'played_at': track.get('added_at', None)
                }
                tracks_data.append(track_data)
        
        # Create DataFrame
        tracks_df = pd.DataFrame(tracks_data)
        
        # Remove duplicates, keeping the highest weight
        tracks_df = tracks_df.sort_values('weight', ascending=False).drop_duplicates('track_id', keep='first')
        
        # Add audio features
        audio_features_df = self._create_audio_features_dataframe(user_data.get('audio_features', []))
        
        if not audio_features_df.empty:
            tracks_df = tracks_df.merge(audio_features_df, on='track_id', how='left')
        
        logger.info(f"Created tracks DataFrame with {len(tracks_df)} unique tracks")
        return tracks_df
    
    def _create_audio_features_dataframe(self, audio_features: List[Dict]) -> pd.DataFrame:
        """Create DataFrame from audio features"""
        if not audio_features:
            return pd.DataFrame()
        
        # Extract relevant audio features
        features_data = []
        for features in audio_features:
            if features:  # Skip None features
                feature_data = {
                    'track_id': features['id'],
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'key': features['key'],
                    'loudness': features['loudness'],
                    'mode': features['mode'],
                    'speechiness': features['speechiness'],
                    'acousticness': features['acousticness'],
                    'instrumentalness': features['instrumentalness'],
                    'liveness': features['liveness'],
                    'valence': features['valence'],
                    'tempo': features['tempo'],
                    'time_signature': features['time_signature']
                }
                features_data.append(feature_data)
        
        return pd.DataFrame(features_data)
    
    def get_user_profile(self, user_data: Dict) -> Dict:
        """Generate user profile from collected data"""
        tracks_df = self.create_tracks_dataframe(user_data)
        
        if tracks_df.empty:
            return {}
        
        # Calculate user preferences
        profile = {
            'total_tracks': len(tracks_df),
            'unique_artists': tracks_df['artist_name'].nunique(),
            'unique_albums': tracks_df['album_name'].nunique(),
            'avg_popularity': tracks_df['popularity'].mean(),
            'explicit_ratio': tracks_df['explicit'].mean(),
        }
        
        # Audio features statistics
        audio_features = ['danceability', 'energy', 'valence', 'acousticness', 
                         'instrumentalness', 'liveness', 'speechiness', 'tempo']
        
        for feature in audio_features:
            if feature in tracks_df.columns:
                profile[f'avg_{feature}'] = tracks_df[feature].mean()
                profile[f'std_{feature}'] = tracks_df[feature].std()
        
        # Top artists
        top_artists = tracks_df['artist_name'].value_counts().head(10).to_dict()
        profile['top_artists'] = top_artists
        
        # Genre analysis (if available)
        if 'album_genres' in tracks_df.columns:
            all_genres = []
            for genres in tracks_df['album_genres'].dropna():
                all_genres.extend(genres)
            profile['top_genres'] = pd.Series(all_genres).value_counts().head(10).to_dict()
        
        logger.info("User profile generated successfully")
        return profile
    
    def get_known_songs(self, user_data: Dict) -> set:
        """Get ALL songs the user already knows (comprehensive filtering)"""
        known_songs = set()
        
        # Add liked songs (highest priority - user explicitly liked these)
        for track in user_data.get('liked_songs', []):
            known_songs.add(track['track_id'])
        
        # Add playlist tracks (user added these to playlists)
        for playlist_id, tracks in user_data.get('playlist_tracks', {}).items():
            for track in tracks:
                known_songs.add(track['track_id'])
        
        # Add tracks from saved albums (user saved entire albums)
        for album_id, tracks in user_data.get('saved_album_tracks', {}).items():
            for track in tracks:
                known_songs.add(track['track_id'])
        
        # Add top tracks (user has listened to these extensively)
        for category in ['top_tracks_short', 'top_tracks_medium', 'top_tracks_long']:
            for track in user_data.get(category, []):
                known_songs.add(track['track_id'])
        
        # Add recently played (user has heard these recently)
        for track in user_data.get('recently_played', []):
            known_songs.add(track['track_id'])
        
        # Log detailed breakdown
        liked_count = len(user_data.get('liked_songs', []))
        playlist_count = sum(len(tracks) for tracks in user_data.get('playlist_tracks', {}).values())
        album_count = sum(len(tracks) for tracks in user_data.get('saved_album_tracks', {}).values())
        top_count = sum(len(user_data.get(category, [])) for category in ['top_tracks_short', 'top_tracks_medium', 'top_tracks_long'])
        recent_count = len(user_data.get('recently_played', []))
        
        logger.info(f"Comprehensive known songs breakdown:")
        logger.info(f"  - Liked songs: {liked_count}")
        logger.info(f"  - Playlist tracks: {playlist_count}")
        logger.info(f"  - Saved album tracks: {album_count}")
        logger.info(f"  - Top tracks: {top_count}")
        logger.info(f"  - Recently played: {recent_count}")
        logger.info(f"  - Total unique known songs: {len(known_songs)}")
        
        return known_songs
    
    def collect_comprehensive_known_songs(self) -> set:
        """Collect ALL known songs using Spotify API directly.

        All helper methods (get_liked_songs, get_recently_played, etc.)
        return *normalised* dicts with a top-level ``track_id`` key,
        so we access ``track['track_id']`` – NOT the raw Spotify nesting.
        """
        if self.spotify_client is None:
            raise ValueError("Spotify client not initialized")
        
        known_songs = set()
        
        try:
            # 1. Liked Songs
            logger.info("Collecting liked songs...")
            liked_songs = self.spotify_client.get_liked_songs()
            for track in liked_songs:
                if track.get('track_id'):
                    known_songs.add(track['track_id'])
            logger.info(f"Found {len(liked_songs)} liked songs")
            
            # 2. Playlist tracks
            logger.info("Collecting playlist tracks...")
            playlists = self.spotify_client.get_user_playlists_comprehensive()
            for playlist in playlists:
                playlist_id = playlist.get('playlist_id')
                if playlist_id:
                    tracks = self.spotify_client.get_playlist_tracks(playlist_id)
                    for track in tracks:
                        if track.get('track_id'):
                            known_songs.add(track['track_id'])
            logger.info(f"Found tracks from {len(playlists)} playlists")
            
            # 3. Recently played
            logger.info("Collecting recently played tracks...")
            recently_played = self.spotify_client.get_recently_played(limit=50)
            for track in recently_played:
                if track.get('track_id'):
                    known_songs.add(track['track_id'])
            logger.info(f"Found {len(recently_played)} recently played tracks")
            
            # 4. Top tracks (short, medium, long term)
            logger.info("Collecting top tracks...")
            for time_range in ['short_term', 'medium_term', 'long_term']:
                top_tracks = self.spotify_client.get_top_tracks(time_range=time_range, limit=50)
                for track in top_tracks:
                    if track.get('track_id'):
                        known_songs.add(track['track_id'])
            logger.info("Collected top tracks from all time ranges")
            
            logger.info(f"TOTAL KNOWN SONGS COLLECTED: {len(known_songs)}")
            return known_songs
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive known songs: {str(e)}")
            return known_songs
