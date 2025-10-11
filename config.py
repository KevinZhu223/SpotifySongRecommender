import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Spotify API credentials
    SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:5001/callback')
    
    # Flask configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # Data storage
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    
    # Recommendation settings
    MAX_RECOMMENDATIONS = 20
    MIN_PLAYS_THRESHOLD = 3
    SIMILARITY_THRESHOLD = 0.7
