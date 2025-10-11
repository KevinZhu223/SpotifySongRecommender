from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
import math

from config import Config
from spotify_client import SpotifyClient
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from recommendation_models import RecommendationModels
from feedback_manager import FeedbackManager
from analytics_manager import AnalyticsManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Global variables for storing data and models
spotify_client = None
data_collector = None
preprocessor = None
recommendation_models = None
feedback_manager = None
analytics_manager = None
user_data = None
tracks_df = None
user_profile = None
known_songs = None
shown_recommendations = set()  # Track recommendations that have been shown

def initialize_components():
    """Initialize all components"""
    global spotify_client, data_collector, preprocessor, recommendation_models, feedback_manager, analytics_manager
    
    try:
        logger.info("Initializing DataCollector...")
        data_collector = DataCollector()
        logger.info("DataCollector initialized")
        
        logger.info("Initializing DataPreprocessor...")
        preprocessor = DataPreprocessor()
        logger.info("DataPreprocessor initialized")
        
        logger.info("Initializing RecommendationModels...")
        recommendation_models = RecommendationModels()
        logger.info("RecommendationModels initialized")
        
        logger.info("Initializing FeedbackManager...")
        feedback_manager = FeedbackManager()
        logger.info("FeedbackManager initialized")
        
        logger.info("Initializing AnalyticsManager...")
        analytics_manager = AnalyticsManager()
        logger.info("AnalyticsManager initialized")
        
        logger.info("Core components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def get_spotify_client():
    """Get or create Spotify client"""
    global spotify_client
    if spotify_client is None:
        try:
            spotify_client = SpotifyClient()
            logger.info("Spotify client initialized")
        except Exception as e:
            logger.error(f"Error initializing Spotify client: {str(e)}")
            return None
    return spotify_client

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/collect_data', methods=['POST'])
def collect_data():
    """Collect user data from Spotify"""
    global user_data, tracks_df, user_profile
    
    try:
        logger.info("Starting data collection...")
        
        # Get Spotify client (will initialize if needed)
        client = get_spotify_client()
        if client is None:
            return jsonify({
                'success': False,
                'message': 'Failed to initialize Spotify client. Please check your credentials.'
            }), 500
        
        # Create a new data collector with the client
        data_collector_with_client = DataCollector(spotify_client=client)
        
        # Collect user data
        user_data = data_collector_with_client.collect_user_data(save_to_file=True)
        
        # Create tracks DataFrame
        tracks_df = data_collector_with_client.create_tracks_dataframe(user_data)
        
        # Preprocess data
        tracks_df = preprocessor.preprocess_tracks_data(tracks_df)
        
        # Create user profile
        user_profile = preprocessor.create_user_profile_vector(tracks_df)
        
        # Get known songs for filtering
        known_songs = data_collector_with_client.get_known_songs(user_data)
        
        # Add feedback-based known songs
        if feedback_manager:
            liked_tracks = feedback_manager.get_liked_tracks()
            disliked_tracks = feedback_manager.get_disliked_tracks()
            known_songs.update(liked_tracks)  # Add liked tracks to known songs
            known_songs.update(disliked_tracks)  # Add disliked tracks to known songs
            logger.info(f"Added {len(liked_tracks)} liked and {len(disliked_tracks)} disliked tracks to known songs")
        
        # Fit recommendation models
        recommendation_models.fit_hybrid_model(tracks_df)
        
        # Save models and preprocessing objects
        recommendation_models.save_models()
        preprocessor.save_preprocessing_objects()
        
        # Get user profile summary
        profile_summary = data_collector_with_client.get_user_profile(user_data)
        
        # Save user data to session for persistence
        session['user_data_loaded'] = True
        session['data_collection_time'] = datetime.now().isoformat()
        
        return jsonify({
            'success': True,
            'message': 'Data collected successfully',
            'profile': profile_summary,
            'tracks_count': len(tracks_df),
            'known_songs_count': len(known_songs)
        })
        
    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error collecting data: {str(e)}'
        }), 500

@app.route('/collect_comprehensive_known_songs', methods=['POST'])
def collect_comprehensive_known_songs():
    """Collect comprehensive known songs using direct Spotify API calls"""
    global known_songs
    
    try:
        # Get Spotify client
        spotify_client = get_spotify_client()
        if not spotify_client:
            return jsonify({
                'success': False,
                'message': 'Spotify client not initialized. Please authenticate first.'
            }), 400
        
        # Create data collector with client
        data_collector = DataCollector(spotify_client)
        
        # Collect comprehensive known songs
        comprehensive_known_songs = data_collector.collect_comprehensive_known_songs()
        
        # Update global known_songs
        known_songs = comprehensive_known_songs
        
        # Add feedback-based known songs
        if feedback_manager:
            liked_tracks = feedback_manager.get_liked_tracks()
            disliked_tracks = feedback_manager.get_disliked_tracks()
            known_songs.update(liked_tracks)
            known_songs.update(disliked_tracks)
            logger.info(f"Added {len(liked_tracks)} liked and {len(disliked_tracks)} disliked tracks to known songs")
        
        return jsonify({
            'success': True,
            'message': 'Comprehensive known songs collected successfully',
            'known_songs_count': len(known_songs)
        })
        
    except Exception as e:
        logger.error(f"Error collecting comprehensive known songs: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error collecting comprehensive known songs: {str(e)}'
        }), 500

@app.route('/load_saved_data', methods=['POST'])
def load_saved_data():
    """Load previously saved user data"""
    global user_data, tracks_df, user_profile, known_songs
    
    try:
        logger.info("Loading saved user data...")
        
        # Try to load the most recent user data file
        data_dir = "data"
        if not os.path.exists(data_dir):
            return jsonify({
                'success': False,
                'message': 'No saved data found'
            }), 404
        
        # Find the most recent user data file
        user_data_files = [f for f in os.listdir(data_dir) if f.startswith('user_data_') and f.endswith('.json')]
        if not user_data_files:
            return jsonify({
                'success': False,
                'message': 'No saved data found'
            }), 404
        
        # Sort by modification time and get the most recent
        user_data_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
        latest_file = os.path.join(data_dir, user_data_files[0])
        
        # Load user data
        with open(latest_file, 'r') as f:
            loaded_user_data = json.load(f)
        
        # Create data collector (without client for loading)
        data_collector_without_client = DataCollector()
        
        # Create tracks DataFrame
        loaded_tracks_df = data_collector_without_client.create_tracks_dataframe(loaded_user_data)
        
        # Preprocess data
        loaded_tracks_df = preprocessor.preprocess_tracks_data(loaded_tracks_df)
        
        # Create user profile
        loaded_user_profile = preprocessor.create_user_profile_vector(loaded_tracks_df)
        
        # Get known songs for filtering
        loaded_known_songs = data_collector_without_client.get_known_songs(loaded_user_data)
        
        # Add feedback-based known songs
        if feedback_manager:
            liked_tracks = feedback_manager.get_liked_tracks()
            disliked_tracks = feedback_manager.get_disliked_tracks()
            loaded_known_songs.update(liked_tracks)
            loaded_known_songs.update(disliked_tracks)
            logger.info(f"Added {len(liked_tracks)} liked and {len(disliked_tracks)} disliked tracks to known songs")
        
        # Try to load existing models
        try:
            recommendation_models.load_models()
            preprocessor.load_preprocessing_objects()
        except:
            # If models don't exist, fit new ones
            recommendation_models.fit_hybrid_model(loaded_tracks_df)
            recommendation_models.save_models()
            preprocessor.save_preprocessing_objects()
        
        # Set global variables
        user_data = loaded_user_data
        tracks_df = loaded_tracks_df
        user_profile = loaded_user_profile
        known_songs = loaded_known_songs
        
        # Mark data as loaded in session
        session['user_data_loaded'] = True
        session['data_collection_time'] = datetime.fromtimestamp(os.path.getmtime(latest_file)).isoformat()
        
        return jsonify({
            'success': True,
            'message': 'Saved data loaded successfully',
            'tracks_count': len(loaded_tracks_df),
            'known_songs_count': len(loaded_known_songs),
            'data_file': user_data_files[0]
        })
        
    except Exception as e:
        logger.error(f"Error loading saved data: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error loading saved data: {str(e)}'
        }), 500

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get recommendations based on different criteria"""
    global tracks_df, user_profile, recommendation_models, known_songs, shown_recommendations
    
    if tracks_df is None or user_profile is None:
        return jsonify({
            'success': False,
            'message': 'No user data available. Please collect data first.'
        }), 400
    
    try:
        recommendation_type = request.args.get('type', 'user_based')
        n_recommendations = int(request.args.get('n', 10))
        track_id = request.args.get('track_id')
        
        recommendations = []
        
        if recommendation_type == 'user_based':
            recommendations = recommendation_models.get_spotify_recommendations_with_fallback(
                tracks_df, known_songs, n_recommendations
            )
        elif recommendation_type == 'content_based' and track_id:
            recommendations = recommendation_models.get_content_based_recommendations(
                track_id, tracks_df, n_recommendations, known_songs
            )
        elif recommendation_type == 'hybrid' and track_id:
            recommendations = recommendation_models.get_hybrid_recommendations(
                track_id, user_profile, tracks_df, n_recommendations, known_songs=known_songs
            )
        elif recommendation_type == 'diverse':
            recommendations = recommendation_models.get_spotify_recommendations_with_fallback(
                tracks_df, known_songs, n_recommendations
            )
        elif recommendation_type == 'spotify':
            recommendations = recommendation_models.get_spotify_recommendations(
                tracks_df, known_songs, n_recommendations
            )
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid recommendation type or missing track_id'
            }), 400
        
        # Filter out previously shown recommendations
        fresh_recommendations = []
        for rec in recommendations:
            if rec['track_id'] not in shown_recommendations:
                fresh_recommendations.append(rec)
                shown_recommendations.add(rec['track_id'])
        
        # If we don't have enough fresh recommendations, add some from the original list
        if len(fresh_recommendations) < n_recommendations:
            for rec in recommendations:
                if len(fresh_recommendations) >= n_recommendations:
                    break
                if rec not in fresh_recommendations:
                    fresh_recommendations.append(rec)
        
        # If still not enough, reset shown recommendations and use all
        if len(fresh_recommendations) < n_recommendations:
            shown_recommendations.clear()
            fresh_recommendations = recommendations[:n_recommendations]
            for rec in fresh_recommendations:
                shown_recommendations.add(rec['track_id'])
        
        # Record analytics
        if analytics_manager:
            analytics_manager.record_recommendation(
                recommendation_type=recommendation_type,
                tracks_recommended=fresh_recommendations,
                user_profile_features=user_profile.tolist() if hasattr(user_profile, 'tolist') else list(user_profile)
            )
        
        return jsonify({
            'success': True,
            'recommendations': fresh_recommendations,
            'type': recommendation_type
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting recommendations: {str(e)}'
        }), 500

@app.route('/search', methods=['GET'])
@app.route('/search_tracks', methods=['GET'])
def search_tracks():
    """Search for tracks"""
    global recommendation_models, tracks_df
    
    if recommendation_models is None:
        return jsonify({
            'success': False,
            'message': 'Models not initialized'
        }), 400
    
    if tracks_df is None:
        return jsonify({
            'success': False,
            'message': 'No user data available. Please collect data first.'
        }), 400
    
    try:
        query = request.args.get('q', '')
        n_results = int(request.args.get('n', 10))
        
        if not query:
            return jsonify({
                'success': False,
                'message': 'Search query is required'
            }), 400
        
        results = recommendation_models.search_similar_tracks(query, tracks_df, n_results)
        
        return jsonify({
            'success': True,
            'results': results,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error searching tracks: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error searching tracks: {str(e)}'
        }), 500

@app.route('/user_tracks', methods=['GET'])
def get_user_tracks():
    """Get user's tracks for selection"""
    global tracks_df
    
    if tracks_df is None:
        return jsonify({
            'success': False,
            'message': 'No user data available'
        }), 400
    
    try:
        # Get top tracks for selection
        top_tracks = tracks_df.nlargest(50, 'weight')[
            ['track_id', 'track_name', 'artist_name', 'album_name', 'popularity']
        ].to_dict('records')
        
        return jsonify({
            'success': True,
            'tracks': top_tracks
        })
        
    except Exception as e:
        logger.error(f"Error getting user tracks: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting user tracks: {str(e)}'
        }), 500

@app.route('/user_profile', methods=['GET'])
def get_user_profile():
    """Get user profile information"""
    global user_data, tracks_df
    
    if user_data is None:
        return jsonify({
            'success': False,
            'message': 'No user data available'
        }), 400
    
    try:
        profile = data_collector.get_user_profile(user_data)
        
        # Add enhanced profile data
        if tracks_df is not None and not tracks_df.empty:
            # Add top songs (most weighted tracks) with listen counts
            if 'weight' in tracks_df.columns:
                top_songs_data = tracks_df.nlargest(10, 'weight')[['track_name', 'artist_name', 'album_name', 'weight']].copy()
                # Convert weight to approximate listen count (weight * 10 for display purposes)
                top_songs_data['listen_count'] = (top_songs_data['weight'] * 10).round().astype(int)
                top_songs = top_songs_data.to_dict('records')
                profile['top_songs'] = top_songs
            
            # Add top artists (by frequency and weight)
            artist_counts = {}
            for _, track in tracks_df.iterrows():
                artist = track['artist_name']
                weight = track.get('weight', 1.0)
                if artist in artist_counts:
                    artist_counts[artist] += weight
                else:
                    artist_counts[artist] = weight
            
            # Sort by weighted count and get top 10
            top_artists = dict(sorted(artist_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            profile['top_artists'] = top_artists
            
            # Add genre information if available
            if 'genres' in tracks_df.columns:
                genre_counts = {}
                for _, track in tracks_df.iterrows():
                    genres = track.get('genres', [])
                    if isinstance(genres, list):
                        for genre in genres:
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
                profile['top_genres'] = dict(sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Add audio feature averages (use engineered features if audio features not available)
            audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
            engineered_features = ['energy_danceability', 'mood_score', 'acoustic_electronic', 'complexity_score']
            
            # Check if we have actual audio features
            has_audio_features = any(feature in tracks_df.columns for feature in audio_features)
            
            if has_audio_features:
                # Use actual audio features
                for feature in audio_features:
                    if feature in tracks_df.columns:
                        profile[f'avg_{feature}'] = tracks_df[feature].mean()
            else:
                # Use engineered features as fallback with proper mapping
                logger.info(f"Available columns in tracks_df: {list(tracks_df.columns)}")
                
                # Map engineered features to audio features
                feature_mapping = {
                    'avg_danceability': 'energy_danceability',
                    'avg_energy': 'energy_danceability', 
                    'avg_valence': 'mood_score',
                    'avg_acousticness': 'acoustic_electronic',
                    'avg_instrumentalness': 'complexity_score'
                }
                
                for audio_feature, engineered_feature in feature_mapping.items():
                    if engineered_feature in tracks_df.columns:
                        value = tracks_df[engineered_feature].mean()
                        if not math.isnan(value) and value is not None:
                            profile[audio_feature] = max(0, min(1, float(value)))
                        else:
                            profile[audio_feature] = 0.5
                    else:
                        profile[audio_feature] = 0.5
                
                logger.info(f"Set audio features from engineered features: {[(k, v) for k, v in profile.items() if k.startswith('avg_')]}")
                logger.info(f"Engineered features available: {[col for col in tracks_df.columns if col in ['energy_danceability', 'mood_score', 'acoustic_electronic', 'complexity_score']]}")
                logger.info(f"Sample engineered feature values: {tracks_df[['energy_danceability', 'mood_score', 'acoustic_electronic', 'complexity_score']].head() if all(col in tracks_df.columns for col in ['energy_danceability', 'mood_score', 'acoustic_electronic', 'complexity_score']) else 'Features not available'}")
        
        return jsonify({
            'success': True,
            'profile': profile
        })
        
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting user profile: {str(e)}'
        }), 500

@app.route('/profile')
def profile_page():
    """Serve the interactive profile page"""
    return render_template('profile.html')

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Handle user feedback (like/dislike)"""
    global feedback_manager
    
    if feedback_manager is None:
        return jsonify({
            'success': False,
            'message': 'Feedback system not initialized'
        }), 500
    
    try:
        data = request.get_json()
        action = data.get('action')  # 'like' or 'dislike'
        track_id = data.get('track_id')
        track_name = data.get('track_name')
        artist_name = data.get('artist_name')
        recommendation_type = data.get('recommendation_type', 'unknown')
        
        if not all([action, track_id, track_name, artist_name]):
            return jsonify({
                'success': False,
                'message': 'Missing required fields'
            }), 400
        
        if action == 'like':
            feedback_manager.add_like(track_id, track_name, artist_name, recommendation_type)
        elif action == 'dislike':
            feedback_manager.add_dislike(track_id, track_name, artist_name, recommendation_type)
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid action. Use "like" or "dislike"'
            }), 400
        
        # Record analytics
        if analytics_manager:
            analytics_manager.record_user_interaction(
                track_id=track_id,
                track_name=track_name,
                artist_name=artist_name,
                action=action,
                recommendation_type=recommendation_type
            )
        
        return jsonify({
            'success': True,
            'message': f'Feedback recorded: {action} for {track_name}'
        })
        
    except Exception as e:
        logger.error(f"Error handling feedback: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error recording feedback: {str(e)}'
        }), 500

@app.route('/feedback/stats', methods=['GET'])
def get_feedback_stats():
    """Get feedback statistics"""
    global feedback_manager
    
    if feedback_manager is None:
        return jsonify({
            'success': False,
            'message': 'Feedback system not initialized'
        }), 500
    
    try:
        stats = feedback_manager.get_feedback_stats()
        history = feedback_manager.get_feedback_history(limit=20)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_feedback': history
        })
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting feedback stats: {str(e)}'
        }), 500

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get recommendation analytics"""
    global analytics_manager
    
    if analytics_manager is None:
        return jsonify({
            'success': False,
            'message': 'Analytics system not initialized'
        }), 500
    
    try:
        days = int(request.args.get('days', 30))
        analytics = analytics_manager.get_recommendation_analytics(days=days)
        trends = analytics_manager.get_trend_analysis(days=days)
        performance = analytics_manager.get_performance_metrics()
        
        return jsonify({
            'success': True,
            'analytics': analytics,
            'trends': trends,
            'performance': performance
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error getting analytics: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'spotify_client': spotify_client is not None,
            'data_collector': data_collector is not None,
            'preprocessor': preprocessor is not None,
            'recommendation_models': recommendation_models is not None
        },
        'user_data_available': user_data is not None,
        'tracks_available': tracks_df is not None
    })

if __name__ == '__main__':
    # Initialize components
    if initialize_components():
        logger.info("Starting Flask application...")
        logger.info("Open http://localhost:5001 in your browser")
        app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5001)
    else:
        logger.error("Failed to initialize components. Exiting...")
