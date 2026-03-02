"""
Spotify Song Recommender – Flask Application
=============================================
Rewritten with:
  * Proper browser-based OAuth (/login  → /callback)
  * Session-based state (no unsafe globals)
  * Tinder-style swipe discovery endpoint  (/discover)
  * One-click playlist export to Spotify   (/export_playlist)
"""

from flask import (
    Flask, render_template, request, jsonify,
    session, redirect, url_for,
)
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
import json
import math

from config import Config
from spotify_client import SpotifyClient, get_oauth_manager
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from recommendation_models import RecommendationModels
from feedback_manager import FeedbackManager
from analytics_manager import AnalyticsManager

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY

# ---------------------------------------------------------------------------
# Lightweight in-process cache  (not globals that break across users)
# In production you'd use Redis / DB.  For a single-user local app this is fine.
# ---------------------------------------------------------------------------
_cache: dict = {}

def _get(key, default=None):
    return _cache.get(key, default)

def _put(key, value):
    _cache[key] = value

# ---------------------------------------------------------------------------
# Component singletons (stateless helpers – safe to share)
# ---------------------------------------------------------------------------
preprocessor = DataPreprocessor()
rec_models = RecommendationModels()
feedback_mgr = FeedbackManager()
analytics_mgr = AnalyticsManager()


def _get_spotify_client() -> SpotifyClient | None:
    """Build a SpotifyClient from the token stored in the Flask session."""
    token_info = session.get("token_info")
    if not token_info:
        return None
    try:
        client = SpotifyClient(token_info=token_info)
        # Persist possibly-refreshed token back into the session
        session["token_info"] = client.token_info
        return client
    except Exception as e:
        logger.error(f"SpotifyClient from session token failed: {e}")
        return None


# ===================================================================
# AUTH ROUTES
# ===================================================================
@app.route("/login")
def login():
    """Redirect browser to Spotify authorisation page."""
    # Clear stale spotipy cache so a fresh token exchange always happens
    if os.path.exists(".cache"):
        os.remove(".cache")
    # Reset the singleton so it doesn't hold stale state
    from spotify_client import reset_oauth_manager
    reset_oauth_manager()

    oauth = get_oauth_manager()
    auth_url = oauth.get_authorize_url()
    return redirect(auth_url)


@app.route("/callback")
def callback():
    """Spotify redirects back here with ?code=..."""
    # Spotify sends ?error=access_denied when the user declines
    error = request.args.get("error")
    if error:
        logger.error(f"Spotify auth error: {error}")
        return redirect(url_for("index"))

    code = request.args.get("code")
    if not code:
        return "Missing authorisation code", 400

    try:
        oauth = get_oauth_manager()
        # check_cache=False ensures we always exchange the fresh auth code
        token_info = oauth.get_access_token(code, check_cache=False)
        session["token_info"] = token_info
        logger.info("OAuth callback: token acquired successfully")
    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        session.pop("token_info", None)
        return redirect(url_for("index"))

    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.clear()
    _cache.clear()
    # Remove spotipy cache file so stale tokens don't linger
    if os.path.exists(".cache"):
        os.remove(".cache")
    from spotify_client import reset_oauth_manager
    reset_oauth_manager()
    return redirect(url_for("index"))


# ===================================================================
# PAGE ROUTES
# ===================================================================
@app.route("/")
def index():
    logged_in = bool(session.get("token_info"))
    has_data = _get("tracks_df") is not None
    return render_template("index.html", logged_in=logged_in, has_data=has_data)


@app.route("/profile")
def profile_page():
    return render_template("profile.html")


@app.route("/discover")
def discover_page():
    """Serve the Tinder-style swipe discovery page."""
    return render_template("discover.html")


# ===================================================================
# DATA COLLECTION
# ===================================================================
@app.route("/collect_data", methods=["POST"])
def collect_data():
    client = _get_spotify_client()
    if not client:
        return jsonify(success=False, message="Not logged in. Please log in first."), 401

    try:
        logger.info("Starting data collection …")
        dc = DataCollector(spotify_client=client)
        user_data = dc.collect_user_data(save_to_file=True)
        tracks_df = dc.create_tracks_dataframe(user_data)
        tracks_df = preprocessor.preprocess_tracks_data(tracks_df)
        user_profile = preprocessor.create_user_profile_vector(tracks_df)
        known_songs = dc.get_known_songs(user_data)

        # Merge feedback
        known_songs.update(feedback_mgr.get_liked_tracks())
        known_songs.update(feedback_mgr.get_disliked_tracks())

        # Fit models
        rec_models.fit_hybrid_model(tracks_df)
        rec_models.save_models()
        preprocessor.save_preprocessing_objects()

        # Store in cache
        _put("user_data", user_data)
        _put("tracks_df", tracks_df)
        _put("user_profile", user_profile)
        _put("known_songs", known_songs)

        profile_summary = dc.get_user_profile(user_data)

        return jsonify(
            success=True,
            message="Data collected successfully",
            profile=profile_summary,
            tracks_count=len(tracks_df),
            known_songs_count=len(known_songs),
        )
    except Exception as e:
        logger.error(f"collect_data error: {e}")
        return jsonify(success=False, message=str(e)), 500


@app.route("/load_saved_data", methods=["POST"])
def load_saved_data():
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            return jsonify(success=False, message="No saved data found"), 404

        files = sorted(
            [f for f in os.listdir(data_dir)
             if f.startswith("user_data_") and f.endswith(".json")],
            key=lambda x: os.path.getmtime(os.path.join(data_dir, x)),
            reverse=True,
        )
        if not files:
            return jsonify(success=False, message="No saved data found"), 404

        latest = os.path.join(data_dir, files[0])
        with open(latest, "r") as f:
            user_data = json.load(f)

        dc = DataCollector()
        tracks_df = dc.create_tracks_dataframe(user_data)
        tracks_df = preprocessor.preprocess_tracks_data(tracks_df)
        user_profile = preprocessor.create_user_profile_vector(tracks_df)
        known_songs = dc.get_known_songs(user_data)
        known_songs.update(feedback_mgr.get_liked_tracks())
        known_songs.update(feedback_mgr.get_disliked_tracks())

        try:
            rec_models.load_models()
            preprocessor.load_preprocessing_objects()
        except Exception:
            rec_models.fit_hybrid_model(tracks_df)
            rec_models.save_models()
            preprocessor.save_preprocessing_objects()

        _put("user_data", user_data)
        _put("tracks_df", tracks_df)
        _put("user_profile", user_profile)
        _put("known_songs", known_songs)

        return jsonify(
            success=True,
            message="Saved data loaded",
            tracks_count=len(tracks_df),
            known_songs_count=len(known_songs),
            data_file=files[0],
        )
    except Exception as e:
        logger.error(f"load_saved_data error: {e}")
        return jsonify(success=False, message=str(e)), 500


# ===================================================================
# RECOMMENDATIONS
# ===================================================================
@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    tracks_df = _get("tracks_df")
    user_profile = _get("user_profile")
    known_songs = _get("known_songs") or set()

    if tracks_df is None or user_profile is None:
        return jsonify(success=False,
                       message="No data loaded. Collect or load data first."), 400

    # Get the session-based Spotify client so rec_models can use it
    client = _get_spotify_client()

    try:
        rtype = request.args.get("type", "user_based")
        n = int(request.args.get("n", 10))
        track_id = request.args.get("track_id")

        recs: list = []
        if rtype == "user_based":
            recs = rec_models.get_spotify_recommendations_with_fallback(
                tracks_df, known_songs, n, spotify_client=client
            )
        elif rtype == "content_based" and track_id:
            recs = rec_models.get_content_based_recommendations(
                track_id, tracks_df, n, known_songs
            )
        elif rtype == "hybrid" and track_id:
            recs = rec_models.get_hybrid_recommendations(
                track_id, user_profile, tracks_df, n, known_songs=known_songs
            )
        elif rtype in ("diverse", "spotify"):
            recs = rec_models.get_spotify_recommendations_with_fallback(
                tracks_df, known_songs, n, spotify_client=client
            )
        else:
            return jsonify(success=False,
                           message="Invalid type or missing track_id"), 400

        # Analytics
        if analytics_mgr:
            analytics_mgr.record_recommendation(
                recommendation_type=rtype,
                tracks_recommended=recs,
                user_profile_features=(
                    user_profile.tolist()
                    if hasattr(user_profile, "tolist") else list(user_profile)
                ),
            )

        return jsonify(success=True, recommendations=recs, type=rtype)
    except Exception as e:
        logger.error(f"recommendations error: {e}")
        return jsonify(success=False, message=str(e)), 500


# ===================================================================
# DISCOVER  (Tinder-style swipe – Concept 3)
# ===================================================================
@app.route("/discover/next", methods=["GET"])
def discover_next():
    """Return the next track to swipe on.

    Uses Spotify recommendations seeded by the user's top tracks,
    filters out known songs + already-swiped songs this session,
    and includes a 30-second preview URL for audio playback.
    """
    tracks_df = _get("tracks_df")
    known_songs: set = _get("known_songs") or set()
    swiped: set = _get("swiped_ids") or set()

    if tracks_df is None:
        return jsonify(success=False,
                       message="No data loaded. Collect data first."), 400

    client = _get_spotify_client()
    if client is None:
        return jsonify(success=False,
                       message="Not logged in."), 401

    try:
        # Build seed from highly-weighted tracks
        if "weight" in tracks_df.columns:
            seeds = tracks_df.nlargest(20, "weight")["track_id"].tolist()
        else:
            seeds = tracks_df.head(20)["track_id"].tolist()

        # ---- Active-learning bias from feedback ----
        target: dict = {}
        liked_history = feedback_mgr.get_feedback_history(limit=50)
        recent_likes = [h for h in liked_history if h["action"] == "like"]
        if recent_likes:
            # Pull audio features from recently liked tracks to bias recs
            liked_ids = [h["track_id"] for h in recent_likes[-10:]]
            feats = client.get_audio_features(liked_ids)
            if feats:
                for key in ("energy", "valence", "danceability",
                            "acousticness", "instrumentalness"):
                    vals = [f[key] for f in feats if key in f]
                    if vals:
                        target[f"target_{key}"] = sum(vals) / len(vals)

        import random
        random.shuffle(seeds)
        seed_batch = seeds[:5]

        candidates = client.get_recommendations(
            seed_batch, limit=50, target_features=target or None,
        )

        skip = known_songs | swiped
        for track in candidates:
            tid = track["track_id"]
            if tid in skip:
                continue
            if not track.get("preview_url"):
                continue  # need audio for the swipe UX
            # Mark as shown
            swiped.add(tid)
            _put("swiped_ids", swiped)
            return jsonify(success=True, track=track)

        # Fallback: try a second batch with different seeds
        random.shuffle(seeds)
        candidates2 = client.get_recommendations(
            seeds[:5], limit=50, target_features=target or None,
        )
        for track in candidates2:
            tid = track["track_id"]
            if tid in skip:
                continue
            if not track.get("preview_url"):
                continue  # need audio for the swipe UX
            swiped.add(tid)
            _put("swiped_ids", swiped)
            return jsonify(success=True, track=track)

        return jsonify(success=False,
                       message="No more fresh tracks with previews available. "
                               "Try resetting your session.")
    except Exception as e:
        logger.error(f"discover/next error: {e}")
        return jsonify(success=False, message=str(e)), 500


@app.route("/discover/swipe", methods=["POST"])
def discover_swipe():
    """Record a swipe (like / dislike) and update active-learning state."""
    data = request.get_json(force=True)
    action = data.get("action")          # "like" or "dislike"
    track = data.get("track", {})
    track_id = track.get("track_id", data.get("track_id"))
    track_name = track.get("track_name", data.get("track_name", ""))
    artist_name = track.get("artist_name", data.get("artist_name", ""))

    if not action or not track_id:
        return jsonify(success=False, message="Missing action or track_id"), 400

    if action == "like":
        feedback_mgr.add_like(track_id, track_name, artist_name, "discover_swipe")
        # Also add to known so we don't recommend again
        ks = _get("known_songs") or set()
        ks.add(track_id)
        _put("known_songs", ks)

        # Store for the playlist export bucket
        liked_queue: list = _get("discover_liked_queue") or []
        liked_queue.append(track)
        _put("discover_liked_queue", liked_queue)

    elif action == "dislike":
        feedback_mgr.add_dislike(track_id, track_name, artist_name, "discover_swipe")
        ks = _get("known_songs") or set()
        ks.add(track_id)
        _put("known_songs", ks)
    else:
        return jsonify(success=False, message="Invalid action"), 400

    if analytics_mgr:
        analytics_mgr.record_user_interaction(
            track_id=track_id,
            track_name=track_name,
            artist_name=artist_name,
            action=action,
            recommendation_type="discover_swipe",
        )

    return jsonify(success=True,
                   message=f"{action} recorded for {track_name}")


@app.route("/discover/liked", methods=["GET"])
def discover_liked():
    """Return the list of tracks liked during the current discover session."""
    liked_queue = _get("discover_liked_queue") or []
    return jsonify(success=True, tracks=liked_queue,
                   count=len(liked_queue))


@app.route("/discover/reset", methods=["POST"])
def discover_reset():
    """Reset the swiped-IDs so the user can get fresh tracks again."""
    _put("swiped_ids", set())
    return jsonify(success=True, message="Swipe history reset.")


# ===================================================================
# EXPORT PLAYLIST (Concept 4)
# ===================================================================
@app.route("/export_playlist", methods=["POST"])
def export_playlist():
    """Create a Spotify playlist from the provided track IDs.

    JSON body:
      { "track_ids": ["id1", "id2", ...],
        "name": "My AI Discovery Mix",    // optional
        "description": "..."              // optional
      }
    """
    client = _get_spotify_client()
    if not client:
        return jsonify(success=False, message="Not logged in."), 401

    data = request.get_json(force=True)
    track_ids: list = data.get("track_ids", [])

    if not track_ids:
        # Fallback: use the discover liked queue
        liked_queue = _get("discover_liked_queue") or []
        track_ids = [t["track_id"] for t in liked_queue if "track_id" in t]

    if not track_ids:
        return jsonify(success=False,
                       message="No tracks to export. Like some songs first!"), 400

    name = data.get("name",
                     f"AI Discovery Mix – {datetime.now().strftime('%b %d %Y')}")
    desc = data.get("description",
                     "Created by Spotify Song Recommender 🎵")

    try:
        playlist_id = client.create_playlist(name=name, description=desc,
                                              public=False)
        client.add_tracks_to_playlist(playlist_id, track_ids)

        return jsonify(
            success=True,
            message=f"Playlist '{name}' created with {len(track_ids)} tracks!",
            playlist_id=playlist_id,
            playlist_url=f"https://open.spotify.com/playlist/{playlist_id}",
        )
    except Exception as e:
        logger.error(f"export_playlist error: {e}")
        return jsonify(success=False, message=str(e)), 500


# ===================================================================
# EXISTING UTILITY ENDPOINTS (search, profile, feedback, health …)
# ===================================================================
@app.route("/search", methods=["GET"])
@app.route("/search_tracks", methods=["GET"])
def search_tracks():
    tracks_df = _get("tracks_df")
    if tracks_df is None:
        return jsonify(success=False,
                       message="No data loaded."), 400
    try:
        query = request.args.get("q", "")
        n = int(request.args.get("n", 10))
        if not query:
            return jsonify(success=False, message="Query required"), 400
        results = rec_models.search_similar_tracks(query, tracks_df, n)
        return jsonify(success=True, results=results, query=query)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


@app.route("/user_tracks", methods=["GET"])
def get_user_tracks():
    tracks_df = _get("tracks_df")
    if tracks_df is None:
        return jsonify(success=False, message="No data loaded"), 400
    top = tracks_df.nlargest(50, "weight")[
        ["track_id", "track_name", "artist_name", "album_name", "popularity"]
    ].to_dict("records")
    return jsonify(success=True, tracks=top)


@app.route("/user_profile", methods=["GET"])
def get_user_profile():
    user_data = _get("user_data")
    tracks_df = _get("tracks_df")
    if user_data is None:
        return jsonify(success=False, message="No data loaded"), 400
    try:
        dc = DataCollector()
        profile = dc.get_user_profile(user_data)
        if tracks_df is not None and not tracks_df.empty:
            if "weight" in tracks_df.columns:
                top_songs = tracks_df.nlargest(10, "weight")[
                    ["track_name", "artist_name", "album_name", "weight"]
                ].copy()
                top_songs["listen_count"] = (top_songs["weight"] * 10).round().astype(int)
                profile["top_songs"] = top_songs.to_dict("records")
            audio_features = ["danceability", "energy", "valence",
                              "acousticness", "instrumentalness",
                              "liveness", "speechiness"]
            for feat in audio_features:
                if feat in tracks_df.columns:
                    val = tracks_df[feat].mean()
                    profile[f"avg_{feat}"] = 0.0 if (val != val) else float(val)  # NaN-safe
        return jsonify(success=True, profile=profile)
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


@app.route("/feedback", methods=["POST"])
def handle_feedback():
    data = request.get_json(force=True)
    action = data.get("action")
    track_id = data.get("track_id")
    track_name = data.get("track_name", "")
    artist_name = data.get("artist_name", "")
    rec_type = data.get("recommendation_type", "unknown")

    if not action or not track_id:
        return jsonify(success=False, message="Missing fields"), 400

    if action == "like":
        feedback_mgr.add_like(track_id, track_name, artist_name, rec_type)
    elif action == "dislike":
        feedback_mgr.add_dislike(track_id, track_name, artist_name, rec_type)
    else:
        return jsonify(success=False, message="Invalid action"), 400

    if analytics_mgr:
        analytics_mgr.record_user_interaction(
            track_id=track_id, track_name=track_name,
            artist_name=artist_name, action=action,
            recommendation_type=rec_type,
        )
    return jsonify(success=True, message=f"{action} recorded for {track_name}")


@app.route("/feedback/stats", methods=["GET"])
def get_feedback_stats():
    stats = feedback_mgr.get_feedback_stats()
    history = feedback_mgr.get_feedback_history(limit=20)
    return jsonify(success=True, stats=stats, recent_feedback=history)


@app.route("/analytics", methods=["GET"])
def get_analytics():
    try:
        days = int(request.args.get("days", 30))
        return jsonify(
            success=True,
            analytics=analytics_mgr.get_recommendation_analytics(days),
            trends=analytics_mgr.get_trend_analysis(days),
            performance=analytics_mgr.get_performance_metrics(),
        )
    except Exception as e:
        return jsonify(success=False, message=str(e)), 500


@app.route("/health")
def health_check():
    return jsonify(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        logged_in="token_info" in session,
        data_loaded=_get("tracks_df") is not None,
    )


# ===================================================================
if __name__ == "__main__":
    logger.info("Starting Flask application …")
    logger.info("Open http://127.0.0.1:5001 in your browser")
    app.run(debug=Config.DEBUG, host="127.0.0.1", port=5001)
