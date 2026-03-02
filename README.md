# Spotify Song Recommender

A sophisticated music discovery platform using machine learning to analyze Spotify listening history and suggest tracks through multiple AI-driven algorithms. Built with Flask, scikit-learn, and the Spotify Web API.

## Key Features

- **Multi-Model Recommendation Engine**:
    - **Hybrid Filtering**: Combines content-based audio features with user-taste profiling.
    - **Content-Based**: Finds similar tracks using 12+ audio dimensions such as danceability and energy.
    - **User-Based Profiling**: Generates recommendations tailored to unique user listening history.
    - **Diverse Clustering**: Provides variety by sampling from different musical clusters.
- **Tinder-Style Discovery Mode**: A dedicated interface for swiping through recommendations, featuring 30-second audio previews and real-time playlist construction.
- **Comprehensive Analytics**: Visualization of music profiles, including average audio features, top artists, and listening trends.
- **Playlist Export**: Seamlessly export liked discoveries into a private playlist on the user's Spotify account.
- **Local Data Persistence**: Caches and loads previously collected Spotify data to minimize API overhead and improve performance.

## Technical Architecture

- **Frontend**: Responsive Bootstrap 5 UI with a custom swipe interface and real-time AJAX updates.
- **Backend**: Flask (Python) with robust session management and a browser-based OAuth 2.0 flow.
- **AI/ML**: 
    - **scikit-learn**: Implements Cosine Similarity, TF-IDF vectorization, SVD dimensionality reduction, and KMeans clustering.
    - **pandas & numpy**: Utilized for feature engineering and data transformation.
- **Integration**: Spotipy library for comprehensive Spotify Web API interaction.

## Quick Start

### 1. Prerequisites
- Python 3.9+
- A [Spotify Developer](https://developer.spotify.com/dashboard) account.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/SpotifySongRecommender.git
cd SpotifySongRecommender

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
1. Go to the Spotify Developer Dashboard and create a new App.
2. Set the Redirect URI to: `http://127.0.0.1:5001/callback`
3. Create a `.env` file in the root directory:
```dotenv
SPOTIFY_CLIENT_ID=your_id_here
SPOTIFY_CLIENT_SECRET=your_secret_here
SPOTIFY_REDIRECT_URI=http://127.0.0.1:5001/callback
SECRET_KEY=your_secret_key
DEBUG=True
```

### 4. Running the Application
```bash
python run.py
```
Access the application at [http://127.0.0.1:5001](http://127.0.0.1:5001).

## Project Structure

- `app.py`: Main Flask application with OAuth flows and API endpoints.
- `recommendation_models.py`: Core logic for Content-Based, Collaborative, and Hybrid AI models.
- `spotify_client.py`: Normalized wrapper for Spotify API interactions.
- `data_collector.py`: Orchestrates comprehensive data fetching from Spotify.
- `data_preprocessor.py`: Handles feature engineering, scaling, and clustering.
- `feedback_manager.py`: Manages user feedback to refine future recommendations.
- `analytics_manager.py`: Tracks and visualizes user interaction trends.

## Contributing
Contributions are welcome. Please submit a Pull Request for any proposed changes.

## License
This project is licensed under the MIT License.
