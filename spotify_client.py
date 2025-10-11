import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import time
from typing import List, Dict, Optional
import logging
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyClient:
    """Handles all Spotify API interactions"""
    
    def __init__(self):
        self.config = Config()
        self.sp = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Spotify API"""
        try:
            scope = "user-read-recently-played user-top-read user-read-playback-state user-library-read playlist-read-private user-read-email"
            
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=self.config.SPOTIFY_CLIENT_ID,
                client_secret=self.config.SPOTIFY_CLIENT_SECRET,
                redirect_uri=self.config.SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=".cache"
            ))
            
            # Test the connection
            user = self.sp.current_user()
            logger.info(f"Successfully authenticated as: {user['display_name']}")
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
    
    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        """Get recently played tracks"""
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            tracks = []
            
            for item in results['items']:
                track_data = {
                    'track_id': item['track']['id'],
                    'track_name': item['track']['name'],
                    'artist_name': ', '.join([artist['name'] for artist in item['track']['artists']]),
                    'album_name': item['track']['album']['name'],
                    'played_at': item['played_at'],
                    'duration_ms': item['track']['duration_ms'],
                    'popularity': item['track']['popularity'],
                    'explicit': item['track']['explicit']
                }
                tracks.append(track_data)
            
            logger.info(f"Retrieved {len(tracks)} recently played tracks")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting recently played tracks: {str(e)}")
            return []
    
    def get_top_tracks(self, time_range: str = 'medium_term', limit: int = 50) -> List[Dict]:
        """Get user's top tracks"""
        try:
            results = self.sp.current_user_top_tracks(time_range=time_range, limit=limit)
            tracks = []
            
            for item in results['items']:
                track_data = {
                    'track_id': item['id'],
                    'track_name': item['name'],
                    'artist_name': ', '.join([artist['name'] for artist in item['artists']]),
                    'album_name': item['album']['name'],
                    'duration_ms': item['duration_ms'],
                    'popularity': item['popularity'],
                    'explicit': item['explicit']
                }
                tracks.append(track_data)
            
            logger.info(f"Retrieved {len(tracks)} top tracks for {time_range}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting top tracks: {str(e)}")
            return []
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Get audio features for multiple tracks"""
        try:
            # Spotify API allows max 100 tracks per request
            all_features = []
            
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                
                try:
                    features = self.sp.audio_features(batch)
                    
                    # Filter out None values (tracks without features)
                    valid_features = [f for f in features if f is not None]
                    all_features.extend(valid_features)
                    
                    logger.info(f"Retrieved audio features for batch {i//100 + 1}: {len(valid_features)} tracks")
                    
                except Exception as batch_error:
                    logger.warning(f"Error getting audio features for batch {i//100 + 1}: {str(batch_error)}")
                    # Continue with next batch
                    continue
                
                # Rate limiting - increase delay to avoid 403 errors
                time.sleep(0.5)
            
            logger.info(f"Retrieved audio features for {len(all_features)} tracks total")
            return all_features
            
        except Exception as e:
            logger.error(f"Error getting audio features: {str(e)}")
            return []
    
    def get_track_info(self, track_ids: List[str]) -> List[Dict]:
        """Get detailed track information"""
        try:
            tracks_info = []
            
            for i in range(0, len(track_ids), 50):  # Spotify allows max 50 tracks per request
                batch = track_ids[i:i+50]
                results = self.sp.tracks(batch)
                
                for track in results['tracks']:
                    if track:  # Skip None tracks
                        track_info = {
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                            'album_name': track['album']['name'],
                            'album_release_date': track['album']['release_date'],
                            'album_genres': track['album'].get('genres', []),
                            'duration_ms': track['duration_ms'],
                            'popularity': track['popularity'],
                            'explicit': track['explicit'],
                            'available_markets': len(track['available_markets'])
                        }
                        tracks_info.append(track_info)
                
                time.sleep(0.1)  # Rate limiting
            
            logger.info(f"Retrieved detailed info for {len(tracks_info)} tracks")
            return tracks_info
            
        except Exception as e:
            logger.error(f"Error getting track info: {str(e)}")
            return []
    
    def search_tracks(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for tracks"""
        try:
            results = self.sp.search(q=query, type='track', limit=limit)
            tracks = []
            
            for item in results['tracks']['items']:
                track_data = {
                    'track_id': item['id'],
                    'track_name': item['name'],
                    'artist_name': ', '.join([artist['name'] for artist in item['artists']]),
                    'album_name': item['album']['name'],
                    'duration_ms': item['duration_ms'],
                    'popularity': item['popularity'],
                    'explicit': item['explicit']
                }
                tracks.append(track_data)
            
            logger.info(f"Found {len(tracks)} tracks for query: {query}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            return []
    
    def get_user_playlists(self) -> List[Dict]:
        """Get user's playlists"""
        try:
            results = self.sp.current_user_playlists()
            playlists = []
            
            for playlist in results['items']:
                playlist_data = {
                    'playlist_id': playlist['id'],
                    'playlist_name': playlist['name'],
                    'owner': playlist['owner']['display_name'],
                    'tracks_count': playlist['tracks']['total'],
                    'public': playlist['public']
                }
                playlists.append(playlist_data)
            
            logger.info(f"Retrieved {len(playlists)} playlists")
            return playlists
            
        except Exception as e:
            logger.error(f"Error getting playlists: {str(e)}")
            return []
    
    def get_liked_songs(self, limit: int = 50) -> List[Dict]:
        """Get user's liked songs"""
        try:
            results = self.sp.current_user_saved_tracks(limit=limit)
            tracks = []
            
            for item in results['items']:
                track = item['track']
                track_data = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                    'album_name': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'explicit': track['explicit'],
                    'added_at': item['added_at']
                }
                tracks.append(track_data)
            
            logger.info(f"Retrieved {len(tracks)} liked songs")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting liked songs: {str(e)}")
            return []
    
    def get_saved_albums(self, limit: int = 50) -> List[Dict]:
        """Get user's saved albums"""
        try:
            results = self.sp.current_user_saved_albums(limit=limit)
            albums = []
            
            for item in results['items']:
                album = item['album']
                album_data = {
                    'album_id': album['id'],
                    'album_name': album['name'],
                    'artist_name': ', '.join([artist['name'] for artist in album['artists']]),
                    'total_tracks': album['total_tracks'],
                    'popularity': album['popularity'],
                    'added_at': item['added_at']
                }
                albums.append(album_data)
            
            logger.info(f"Retrieved {len(albums)} saved albums")
            return albums
            
        except Exception as e:
            logger.error(f"Error getting saved albums: {str(e)}")
            return []
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Get all tracks from a specific playlist"""
        try:
            results = self.sp.playlist_tracks(playlist_id)
            tracks = []
            
            for item in results['items']:
                if item['track'] and item['track']['type'] == 'track':
                    track = item['track']
                    track_data = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                        'album_name': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'popularity': track['popularity'],
                        'explicit': track['explicit'],
                        'added_at': item['added_at']
                    }
                    tracks.append(track_data)
            
            logger.info(f"Retrieved {len(tracks)} tracks from playlist {playlist_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {str(e)}")
            return []
    
    def get_recommendations(self, seed_tracks: List[str], limit: int = 20) -> List[Dict]:
        """Get recommendations from Spotify's recommendation API"""
        try:
            # Limit to 5 seed tracks (Spotify API limit)
            seed_tracks = seed_tracks[:5]
            
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks,
                limit=limit,
                target_energy=0.5,
                target_valence=0.5,
                target_danceability=0.5
            )
            
            tracks = []
            for track in recommendations['tracks']:
                track_data = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                    'album_name': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity'],
                    'explicit': track['explicit']
                }
                tracks.append(track_data)
            
            logger.info(f"Retrieved {len(tracks)} recommendations from Spotify API")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting Spotify recommendations: {str(e)}")
            return []
    
    def get_user_saved_tracks_comprehensive(self, limit: int = 1000) -> List[Dict]:
        """Get all user's saved tracks (liked songs) with pagination"""
        try:
            all_tracks = []
            offset = 0
            
            while True:
                results = self.sp.current_user_saved_tracks(limit=50, offset=offset)
                
                if not results['items']:
                    break
                
                for item in results['items']:
                    track = item['track']
                    track_data = {
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                        'album_name': track['album']['name'],
                        'duration_ms': track['duration_ms'],
                        'popularity': track['popularity'],
                        'explicit': track['explicit'],
                        'added_at': item['added_at']
                    }
                    all_tracks.append(track_data)
                
                offset += 50
                
                # Break if we've reached the limit or no more tracks
                if len(all_tracks) >= limit or len(results['items']) < 50:
                    break
                
                # Rate limiting
                time.sleep(0.1)
            
            logger.info(f"Retrieved {len(all_tracks)} liked songs (comprehensive)")
            return all_tracks
            
        except Exception as e:
            logger.error(f"Error getting comprehensive liked songs: {str(e)}")
            return []
    
    def get_album_tracks(self, album_id: str) -> List[Dict]:
        """Get all tracks from a specific album"""
        try:
            results = self.sp.album_tracks(album_id)
            tracks = []
            
            for track in results['items']:
                track_data = {
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'artist_name': ', '.join([artist['name'] for artist in track['artists']]),
                    'album_name': '',  # Will be filled from album info
                    'duration_ms': track['duration_ms'],
                    'popularity': 0,  # Not available in album tracks
                    'explicit': track['explicit']
                }
                tracks.append(track_data)
            
            logger.info(f"Retrieved {len(tracks)} tracks from album {album_id}")
            return tracks
            
        except Exception as e:
            logger.error(f"Error getting album tracks: {str(e)}")
            return []
    
    def get_user_saved_albums_comprehensive(self, limit: int = 1000) -> List[Dict]:
        """Get all user's saved albums with pagination"""
        try:
            all_albums = []
            offset = 0
            
            while True:
                results = self.sp.current_user_saved_albums(limit=50, offset=offset)
                
                if not results['items']:
                    break
                
                for item in results['items']:
                    album = item['album']
                    album_data = {
                        'album_id': album['id'],
                        'album_name': album['name'],
                        'artist_name': ', '.join([artist['name'] for artist in album['artists']]),
                        'total_tracks': album['total_tracks'],
                        'popularity': album['popularity'],
                        'added_at': item['added_at']
                    }
                    all_albums.append(album_data)
                
                offset += 50
                
                # Break if we've reached the limit or no more albums
                if len(all_albums) >= limit or len(results['items']) < 50:
                    break
                
                # Rate limiting
                time.sleep(0.1)
            
            logger.info(f"Retrieved {len(all_albums)} saved albums (comprehensive)")
            return all_albums
            
        except Exception as e:
            logger.error(f"Error getting comprehensive saved albums: {str(e)}")
            return []
    
    def get_user_playlists_comprehensive(self, limit: int = 1000) -> List[Dict]:
        """Get all user's playlists with pagination"""
        try:
            all_playlists = []
            offset = 0
            
            while True:
                results = self.sp.current_user_playlists(limit=50, offset=offset)
                
                if not results['items']:
                    break
                
                for playlist in results['items']:
                    playlist_data = {
                        'playlist_id': playlist['id'],
                        'playlist_name': playlist['name'],
                        'owner': playlist['owner']['display_name'],
                        'tracks_count': playlist['tracks']['total'],
                        'public': playlist['public']
                    }
                    all_playlists.append(playlist_data)
                
                offset += 50
                
                # Break if we've reached the limit or no more playlists
                if len(all_playlists) >= limit or len(results['items']) < 50:
                    break
                
                # Rate limiting
                time.sleep(0.1)
            
            logger.info(f"Retrieved {len(all_playlists)} playlists (comprehensive)")
            return all_playlists
            
        except Exception as e:
            logger.error(f"Error getting comprehensive playlists: {str(e)}")
            return []
