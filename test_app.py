#!/usr/bin/env python3
"""
Test script for Spotify Song Recommender
This script tests the basic functionality without requiring Spotify API credentials
"""

import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import Config
        from data_preprocessor import DataPreprocessor
        from recommendation_models import RecommendationModels
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from config import Config
        config = Config()
        
        # Test that config attributes exist
        assert hasattr(config, 'SPOTIFY_CLIENT_ID')
        assert hasattr(config, 'SPOTIFY_CLIENT_SECRET')
        assert hasattr(config, 'DATA_DIR')
        assert hasattr(config, 'MODELS_DIR')
        
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_data_preprocessor():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessor...")
    
    try:
        from data_preprocessor import DataPreprocessor
        
        # Create sample data
        sample_data = {
            'track_id': ['track1', 'track2', 'track3'],
            'track_name': ['Song 1', 'Song 2', 'Song 3'],
            'artist_name': ['Artist 1', 'Artist 2', 'Artist 3'],
            'danceability': [0.8, 0.6, 0.9],
            'energy': [0.7, 0.5, 0.8],
            'valence': [0.6, 0.4, 0.7],
            'acousticness': [0.3, 0.8, 0.2],
            'popularity': [80, 60, 90],
            'duration_ms': [180000, 200000, 160000],
            'explicit': [False, True, False],
            'weight': [1.0, 2.0, 1.5]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test preprocessor
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess_tracks_data(df)
        
        # Check that preprocessing worked
        assert len(processed_df) == 3
        assert 'duration_minutes' in processed_df.columns
        assert 'energy_danceability' in processed_df.columns
        assert 'mood_score' in processed_df.columns
        
        # Test user profile creation
        user_profile = preprocessor.create_user_profile_vector(processed_df)
        assert len(user_profile) > 0
        
        print("✓ Data preprocessor working correctly")
        return True
    except Exception as e:
        print(f"✗ Data preprocessor error: {e}")
        return False

def test_recommendation_models():
    """Test recommendation models"""
    print("\nTesting recommendation models...")
    
    try:
        from recommendation_models import RecommendationModels
        
        # Create sample data
        sample_data = {
            'track_id': ['track1', 'track2', 'track3', 'track4', 'track5'],
            'track_name': ['Song 1', 'Song 2', 'Song 3', 'Song 4', 'Song 5'],
            'artist_name': ['Artist 1', 'Artist 2', 'Artist 3', 'Artist 4', 'Artist 5'],
            'album_name': ['Album 1', 'Album 2', 'Album 3', 'Album 4', 'Album 5'],
            'danceability': [0.8, 0.6, 0.9, 0.7, 0.5],
            'energy': [0.7, 0.5, 0.8, 0.6, 0.4],
            'valence': [0.6, 0.4, 0.7, 0.5, 0.3],
            'acousticness': [0.3, 0.8, 0.2, 0.6, 0.9],
            'instrumentalness': [0.1, 0.2, 0.05, 0.15, 0.3],
            'liveness': [0.2, 0.1, 0.3, 0.25, 0.1],
            'speechiness': [0.05, 0.1, 0.03, 0.08, 0.15],
            'tempo': [120, 100, 140, 110, 90],
            'loudness': [-5, -8, -3, -6, -10],
            'key': [0, 1, 2, 0, 3],
            'mode': [1, 0, 1, 1, 0],
            'time_signature': [4, 4, 4, 3, 4],
            'popularity': [80, 60, 90, 70, 50],
            'duration_ms': [180000, 200000, 160000, 190000, 210000],
            'explicit': [False, True, False, True, False],
            'weight': [1.0, 2.0, 1.5, 1.2, 0.8],
            'source': ['recent', 'top_short', 'top_medium', 'top_long', 'recent']
        }
        
        df = pd.DataFrame(sample_data)
        
        # Test recommendation models
        models = RecommendationModels()
        models.fit_content_based_model(df)
        
        # Test content-based recommendations
        recommendations = models.get_content_based_recommendations('track1', 3)
        assert len(recommendations) <= 3
        assert all('track_id' in rec for rec in recommendations)
        
        # Test user-based recommendations
        user_profile = np.array([0.7, 0.6, 0.5, 0.4, 0.1, 0.2, 0.05, 120, -6, 1, 1, 4])
        user_recs = models.get_user_based_recommendations(user_profile, 3)
        assert len(user_recs) <= 3
        
        print("✓ Recommendation models working correctly")
        return True
    except Exception as e:
        print(f"✗ Recommendation models error: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = ['data', 'models', 'static', 'templates', 'static/css', 'static/js']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ directory exists")
        else:
            print(f"✗ {dir_name}/ directory missing")
            return False
    
    return True

def test_required_files():
    """Test that required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        'app.py',
        'config.py',
        'spotify_client.py',
        'data_collector.py',
        'data_preprocessor.py',
        'recommendation_models.py',
        'requirements.txt',
        'README.md',
        'templates/base.html',
        'templates/index.html',
        'static/css/style.css',
        'static/js/main.js'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ {file_name} exists")
        else:
            print(f"✗ {file_name} missing")
            return False
    
    return True

def main():
    """Run all tests"""
    print("Spotify Song Recommender - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_data_preprocessor,
        test_recommendation_models,
        test_directory_structure,
        test_required_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application is ready to use.")
        print("\nNext steps:")
        print("1. Set up your Spotify API credentials in .env file")
        print("2. Run: python app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
