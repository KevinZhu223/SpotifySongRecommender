"""
Analytics Manager for Music Recommendation Insights
Provides analytics on user behavior, trends, and recommendation performance
"""

import json
import os
import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class AnalyticsManager:
    """Manages analytics and trend analysis for the recommendation system"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.analytics_file = os.path.join(data_dir, "analytics.json")
        self.analytics_data = self._load_analytics()
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def _load_analytics(self) -> Dict:
        """Load analytics data from file"""
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'recommendation_history': [],
                    'user_interactions': [],
                    'trend_data': {},
                    'performance_metrics': {}
                }
        except Exception as e:
            logger.error(f"Error loading analytics data: {str(e)}")
            return {
                'recommendation_history': [],
                'user_interactions': [],
                'trend_data': {},
                'performance_metrics': {}
            }
    
    def _save_analytics(self):
        """Save analytics data to file"""
        try:
            with open(self.analytics_file, 'w') as f:
                json.dump(self.analytics_data, f, indent=2)
            logger.info("Analytics data saved successfully")
        except Exception as e:
            logger.error(f"Error saving analytics data: {str(e)}")
    
    def record_recommendation(self, recommendation_type: str, tracks_recommended: List[Dict], 
                            user_profile_features: List[float], timestamp: str = None):
        """Record a recommendation session"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        session_data = {
            'timestamp': timestamp,
            'recommendation_type': recommendation_type,
            'tracks_count': len(tracks_recommended),
            'user_profile_features': user_profile_features,
            'tracks': [
                {
                    'track_id': track.get('track_id'),
                    'track_name': track.get('track_name'),
                    'artist_name': track.get('artist_name'),
                    'similarity_score': track.get('similarity_score', 0)
                }
                for track in tracks_recommended
            ]
        }
        
        self.analytics_data['recommendation_history'].append(session_data)
        
        # Keep only last 1000 sessions to prevent file from growing too large
        if len(self.analytics_data['recommendation_history']) > 1000:
            self.analytics_data['recommendation_history'] = self.analytics_data['recommendation_history'][-1000:]
        
        self._save_analytics()
        logger.info(f"Recorded recommendation session: {recommendation_type} with {len(tracks_recommended)} tracks")
    
    def record_user_interaction(self, track_id: str, track_name: str, artist_name: str, 
                              action: str, recommendation_type: str, timestamp: str = None):
        """Record user interaction (like/dislike)"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        interaction_data = {
            'timestamp': timestamp,
            'track_id': track_id,
            'track_name': track_name,
            'artist_name': artist_name,
            'action': action,  # 'like' or 'dislike'
            'recommendation_type': recommendation_type
        }
        
        self.analytics_data['user_interactions'].append(interaction_data)
        
        # Keep only last 5000 interactions
        if len(self.analytics_data['user_interactions']) > 5000:
            self.analytics_data['user_interactions'] = self.analytics_data['user_interactions'][-5000:]
        
        self._save_analytics()
        logger.info(f"Recorded user interaction: {action} for {track_name}")
    
    def get_recommendation_analytics(self, days: int = 30) -> Dict:
        """Get analytics on recommendation performance"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # Filter recent data
        recent_recommendations = [
            rec for rec in self.analytics_data['recommendation_history']
            if rec['timestamp'] >= cutoff_str
        ]
        
        recent_interactions = [
            interaction for interaction in self.analytics_data['user_interactions']
            if interaction['timestamp'] >= cutoff_str
        ]
        
        # Calculate metrics
        total_recommendations = len(recent_recommendations)
        total_interactions = len(recent_interactions)
        
        # Recommendation type distribution
        rec_type_counts = {}
        for rec in recent_recommendations:
            rec_type = rec['recommendation_type']
            rec_type_counts[rec_type] = rec_type_counts.get(rec_type, 0) + 1
        
        # Interaction rates
        likes = len([i for i in recent_interactions if i['action'] == 'like'])
        dislikes = len([i for i in recent_interactions if i['action'] == 'dislike'])
        
        # Average similarity scores by recommendation type
        avg_scores = {}
        for rec_type in rec_type_counts.keys():
            scores = []
            for rec in recent_recommendations:
                if rec['recommendation_type'] == rec_type:
                    scores.extend([track['similarity_score'] for track in rec['tracks']])
            if scores:
                avg_scores[rec_type] = np.mean(scores)
        
        return {
            'period_days': days,
            'total_recommendations': total_recommendations,
            'total_interactions': total_interactions,
            'recommendation_type_distribution': rec_type_counts,
            'interaction_rates': {
                'likes': likes,
                'dislikes': dislikes,
                'like_rate': likes / total_interactions if total_interactions > 0 else 0,
                'dislike_rate': dislikes / total_interactions if total_interactions > 0 else 0
            },
            'average_similarity_scores': avg_scores
        }
    
    def get_trend_analysis(self, days: int = 30) -> Dict:
        """Get trend analysis over time"""
        cutoff_date = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # Filter recent data
        recent_recommendations = [
            rec for rec in self.analytics_data['recommendation_history']
            if rec['timestamp'] >= cutoff_str
        ]
        
        recent_interactions = [
            interaction for interaction in self.analytics_data['user_interactions']
            if interaction['timestamp'] >= cutoff_str
        ]
        
        # Daily trends
        daily_trends = {}
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).date()
            date_str = date.isoformat()
            
            daily_recs = [rec for rec in recent_recommendations if rec['timestamp'].startswith(date_str)]
            daily_interactions = [interaction for interaction in recent_interactions if interaction['timestamp'].startswith(date_str)]
            
            daily_trends[date_str] = {
                'recommendations': len(daily_recs),
                'interactions': len(daily_interactions),
                'likes': len([i for i in daily_interactions if i['action'] == 'like']),
                'dislikes': len([i for i in daily_interactions if i['action'] == 'dislike'])
            }
        
        # Artist popularity trends
        artist_interactions = {}
        for interaction in recent_interactions:
            artist = interaction['artist_name']
            if artist not in artist_interactions:
                artist_interactions[artist] = {'likes': 0, 'dislikes': 0}
            artist_interactions[artist][interaction['action'] + 's'] += 1
        
        # Top artists by interaction
        top_artists = sorted(
            artist_interactions.items(),
            key=lambda x: x[1]['likes'] + x[1]['dislikes'],
            reverse=True
        )[:10]
        
        return {
            'period_days': days,
            'daily_trends': daily_trends,
            'top_artists_by_interaction': top_artists,
            'total_artists_interacted': len(artist_interactions)
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get overall performance metrics"""
        total_recommendations = len(self.analytics_data['recommendation_history'])
        total_interactions = len(self.analytics_data['user_interactions'])
        
        if total_recommendations == 0:
            return {
                'total_sessions': 0,
                'total_interactions': 0,
                'interaction_rate': 0,
                'average_tracks_per_session': 0
            }
        
        # Calculate average tracks per session
        total_tracks = sum(rec['tracks_count'] for rec in self.analytics_data['recommendation_history'])
        avg_tracks_per_session = total_tracks / total_recommendations
        
        # Interaction rate
        interaction_rate = total_interactions / total_tracks if total_tracks > 0 else 0
        
        return {
            'total_sessions': total_recommendations,
            'total_interactions': total_interactions,
            'interaction_rate': interaction_rate,
            'average_tracks_per_session': avg_tracks_per_session,
            'total_tracks_recommended': total_tracks
        }
