#!/usr/bin/env python3
"""
Setup script for Spotify Song Recommender
This script helps with initial setup and configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("=" * 60)
    print("Spotify Song Recommender Setup")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error installing dependencies: {e}")
        return False

def create_env_file():
    """Create .env file from template"""
    print("\nSetting up environment configuration...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print(".env file already exists")
        return True
    
    if not env_example.exists():
        print("ERROR: env_example.txt not found")
        return False
    
    # Copy example file
    shutil.copy(env_example, env_file)
    print("Created .env file from template")
    
    print("\nIMPORTANT: You need to configure your Spotify API credentials:")
    print("1. Go to https://developer.spotify.com/dashboard")
    print("2. Create a new app")
    print("3. Copy your Client ID and Client Secret")
    print("4. Edit the .env file and replace the placeholder values")
    print("5. Set the redirect URI to: http://localhost:5001/callback")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = ['data', 'models', 'static/css', 'static/js', 'templates']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created {directory}/")
    
    return True

def run_tests():
    """Run the test suite"""
    print("\nRunning tests...")
    try:
        result = subprocess.run([sys.executable, "test_app.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("All tests passed")
            return True
        else:
            print("Some tests failed")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"ERROR: Error running tests: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Edit the .env file with your Spotify API credentials")
    print("2. Run the application: python app.py")
    print("3. Open http://localhost:5001 in your browser")
    print("4. Click 'Collect My Data' to start using the recommender")
    print("\nFor detailed instructions, see the README.md file")
    print("\nHappy music discovery!")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but setup can continue")
        print("You may need to fix issues before running the application")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
