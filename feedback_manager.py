"""
Feedback Manager for User Preferences
Handles like/dislike feedback for recommendations
"""

import json
import os
import logging
from typing import Dict, List, Set
from datetime import datetime

logger = logging.getLogger(__name__)

class FeedbackManager:
    """Manages user feedback for recommendations"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "user_feedback.json")
        self.feedback_data = self._load_feedback()
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def _load_feedback(self) -> Dict:
        """Load feedback data from file"""
        try:
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    'liked_tracks': [],
                    'disliked_tracks': [],
                    'feedback_history': []
                }
        except Exception as e:
            logger.error(f"Error loading feedback data: {str(e)}")
            return {
                'liked_tracks': [],
                'disliked_tracks': [],
                'feedback_history': []
            }
    
    def _save_feedback(self):
        """Save feedback data to file"""
        try:
            with open(self.feedback_file, 'w') as f:
                json.dump(self.feedback_data, f, indent=2)
            logger.info("Feedback data saved successfully")
        except Exception as e:
            logger.error(f"Error saving feedback data: {str(e)}")
    
    def add_like(self, track_id: str, track_name: str, artist_name: str, recommendation_type: str):
        """Add a liked track"""
        if track_id not in self.feedback_data['liked_tracks']:
            self.feedback_data['liked_tracks'].append(track_id)
            
            # Add to feedback history
            self.feedback_data['feedback_history'].append({
                'track_id': track_id,
                'track_name': track_name,
                'artist_name': artist_name,
                'action': 'like',
                'recommendation_type': recommendation_type,
                'timestamp': datetime.now().isoformat()
            })
            
            # Remove from disliked if it was there
            if track_id in self.feedback_data['disliked_tracks']:
                self.feedback_data['disliked_tracks'].remove(track_id)
            
            self._save_feedback()
            logger.info(f"Added like for track: {track_name} by {artist_name}")
    
    def add_dislike(self, track_id: str, track_name: str, artist_name: str, recommendation_type: str):
        """Add a disliked track"""
        if track_id not in self.feedback_data['disliked_tracks']:
            self.feedback_data['disliked_tracks'].append(track_id)
            
            # Add to feedback history
            self.feedback_data['feedback_history'].append({
                'track_id': track_id,
                'track_name': track_name,
                'artist_name': artist_name,
                'action': 'dislike',
                'recommendation_type': recommendation_type,
                'timestamp': datetime.now().isoformat()
            })
            
            # Remove from liked if it was there
            if track_id in self.feedback_data['liked_tracks']:
                self.feedback_data['liked_tracks'].remove(track_id)
            
            self._save_feedback()
            logger.info(f"Added dislike for track: {track_name} by {artist_name}")
    
    def get_liked_tracks(self) -> Set[str]:
        """Get set of liked track IDs"""
        return set(self.feedback_data['liked_tracks'])
    
    def get_disliked_tracks(self) -> Set[str]:
        """Get set of disliked track IDs"""
        return set(self.feedback_data['disliked_tracks'])
    
    def get_feedback_history(self, limit: int = 50) -> List[Dict]:
        """Get recent feedback history"""
        return self.feedback_data['feedback_history'][-limit:]
    
    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics"""
        return {
            'total_likes': len(self.feedback_data['liked_tracks']),
            'total_dislikes': len(self.feedback_data['disliked_tracks']),
            'total_feedback': len(self.feedback_data['feedback_history'])
        }
    
    def clear_feedback(self):
        """Clear all feedback data"""
        self.feedback_data = {
            'liked_tracks': [],
            'disliked_tracks': [],
            'feedback_history': []
        }
        self._save_feedback()
        logger.info("All feedback data cleared")
