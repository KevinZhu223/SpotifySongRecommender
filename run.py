#!/usr/bin/env python3
"""
Simple launcher script for Spotify Song Recommender
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Check if .env file exists and is configured"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("ERROR: .env file not found!")
        print("Please run 'python setup.py' first to set up the application")
        return False
    
    # Check if credentials are configured
    with open(env_file, 'r') as f:
        content = f.read()
        
    if 'your_client_id_here' in content or 'your_client_secret_here' in content:
        print("ERROR: Spotify API credentials not configured!")
        print("Please edit the .env file with your Spotify API credentials")
        print("Get them from: https://developer.spotify.com/dashboard")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("Starting Spotify Song Recommender...")
    print("=" * 50)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("ERROR: app.py not found!")
        print("Please make sure you're in the correct directory")
        sys.exit(1)
    
    print("Environment check passed")
    print("Starting Flask application...")
    print("Open http://127.0.0.1:5001 in your browser")
    print("=" * 50)
    
    try:
        # Start the Flask application
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"ERROR: Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
