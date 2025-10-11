# Spotify Song Recommender

A machine learning-powered music recommendation system that analyzes your Spotify listening history to suggest new songs you'll love. Built with Flask, scikit-learn, and the Spotify Web API.

## Features

- **Personalized Recommendations**: Get song recommendations based on your listening history
- **Multiple Algorithms**: Content-based filtering, collaborative filtering, and hybrid approaches
- **Audio Feature Analysis**: Uses Spotify's audio features (danceability, energy, valence, etc.)
- **Interactive Web Interface**: Clean, responsive web interface for easy interaction
- **User Profile Analytics**: View detailed insights about your music taste
- **Search Functionality**: Search through your music library
- **Diverse Recommendations**: Get recommendations from different musical clusters

## How It Works

1. **Data Collection**: Connects to your Spotify account to collect:
   - Recently played tracks
   - Top tracks (short, medium, and long term)
   - Audio features for each track
   - Playlist information

2. **Data Processing**: 
   - Preprocesses and normalizes audio features
   - Creates user profile vectors
   - Performs dimensionality reduction (PCA)
   - Clusters tracks for diverse recommendations

3. **Recommendation Algorithms**:
   - **Content-Based**: Finds songs similar to ones you like based on audio features
   - **User-Based**: Recommends songs based on your overall music taste profile
   - **Hybrid**: Combines content-based and user-based approaches
   - **Diverse**: Provides recommendations from different musical clusters

## Installation

### Prerequisites

- Python 3.8 or higher
- Spotify Developer Account
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SpotifySongRecommender
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Spotify API credentials**:
   - Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
   - Create a new app
   - Copy your Client ID and Client Secret
   - Set the redirect URI to `http://localhost:5001/callback`

5. **Create environment file**:
   ```bash
   # Copy the example file
   copy env_example.txt .env
   
   # Edit .env with your credentials
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   SPOTIFY_REDIRECT_URI=http://localhost:5001/callback
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://localhost:5001`

3. **Collect your data**:
   - Click "Collect My Data" to gather your Spotify listening history
   - This will redirect you to Spotify for authentication
   - The process may take a few minutes depending on your library size

4. **Get recommendations**:
   - Choose a recommendation type (User-Based, Content-Based, Hybrid, or Diverse)
   - For Content-Based and Hybrid, select a track from your library
   - Click "Get Recommendations" to see your personalized suggestions

5. **Explore your profile**:
   - Click "View Profile" to see detailed analytics about your music taste
   - View your top artists, audio feature preferences, and listening statistics

## API Endpoints

- `GET /` - Main application interface
- `POST /collect_data` - Collect user data from Spotify
- `GET /recommendations` - Get recommendations
- `GET /search` - Search tracks
- `GET /user_tracks` - Get user's tracks
- `GET /user_profile` - Get user profile
- `GET /health` - Health check

## Configuration

The application can be configured through the `config.py` file:

- `MAX_RECOMMENDATIONS`: Maximum number of recommendations to return
- `MIN_PLAYS_THRESHOLD`: Minimum plays threshold for track inclusion
- `SIMILARITY_THRESHOLD`: Similarity threshold for recommendations

## Data Storage

- User data is stored in JSON format in the `data/` directory
- Trained models are saved in the `models/` directory
- All data is stored locally and not shared with external services

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   - Verify your Spotify API credentials
   - Ensure the redirect URI matches exactly
   - Check that your Spotify app is not in development mode restrictions

2. **No Recommendations**:
   - Ensure you have collected data first
   - Check that you have enough listening history
   - Try different recommendation types

3. **Import Errors**:
   - Make sure all dependencies are installed
   - Check your Python version (3.8+ required)

### Debug Mode

Enable debug mode by setting `DEBUG=True` in your `.env` file for detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Spotify Web API for providing access to music data
- scikit-learn for machine learning algorithms
- Flask for the web framework
- Bootstrap for the UI components

## Future Enhancements

- [ ] Real-time recommendation updates
- [ ] Social features (compare taste with friends)
- [ ] Playlist generation
- [ ] Mood-based recommendations
- [ ] Integration with other music services
- [ ] Mobile app
- [ ] Advanced visualization of music taste evolution

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information

---

**Note**: This application requires a Spotify Premium account for full functionality, as some features require access to user's listening history.
