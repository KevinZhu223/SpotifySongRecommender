import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
from typing import List, Dict, Optional
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared OAuth manager – one per process, re-used across requests
# ---------------------------------------------------------------------------
_oauth_manager: Optional[SpotifyOAuth] = None

SCOPES = (
    "user-read-recently-played "
    "user-top-read "
    "user-read-playback-state "
    "user-library-read "
    "playlist-read-private "
    "playlist-modify-public "
    "playlist-modify-private "
    "user-read-email"
)


def get_oauth_manager() -> SpotifyOAuth:
    """Return (and lazily create) a singleton SpotifyOAuth manager."""
    global _oauth_manager
    if _oauth_manager is None:
        cfg = Config()
        _oauth_manager = SpotifyOAuth(
            client_id=cfg.SPOTIFY_CLIENT_ID,
            client_secret=cfg.SPOTIFY_CLIENT_SECRET,
            redirect_uri=cfg.SPOTIFY_REDIRECT_URI,
            scope=SCOPES,
            cache_path=".cache",
            show_dialog=True,
        )
    return _oauth_manager


def reset_oauth_manager() -> None:
    """Reset the singleton so a fresh SpotifyOAuth is created on next call."""
    global _oauth_manager
    _oauth_manager = None


class SpotifyClient:
    """Handles all Spotify API interactions.

    Construction modes:
      * ``SpotifyClient()`` – tries the cached token on disk (.cache)
      * ``SpotifyClient(token_info={...})`` – uses an explicit token dict
        (the preferred path when the Flask OAuth callback provides a token)
    """

    def __init__(self, token_info: Optional[dict] = None):
        self.config = Config()
        self.sp: Optional[spotipy.Spotify] = None
        self._token_info = token_info
        self._authenticate()

    # ------------------------------------------------------------------
    def _authenticate(self):
        try:
            oauth = get_oauth_manager()
            if self._token_info:
                if oauth.is_token_expired(self._token_info):
                    self._token_info = oauth.refresh_access_token(
                        self._token_info["refresh_token"]
                    )
                self.sp = spotipy.Spotify(auth=self._token_info["access_token"])
            else:
                token_info = oauth.get_cached_token()
                if token_info:
                    if oauth.is_token_expired(token_info):
                        token_info = oauth.refresh_access_token(
                            token_info["refresh_token"]
                        )
                    self._token_info = token_info
                    self.sp = spotipy.Spotify(auth=token_info["access_token"])
                else:
                    raise RuntimeError(
                        "No Spotify token available. Please log in via the web UI."
                    )
            user = self.sp.current_user()
            logger.info(f"Authenticated as: {user['display_name']}")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    @property
    def token_info(self) -> Optional[dict]:
        return self._token_info

    def current_user_id(self) -> str:
        return self.sp.current_user()["id"]

    # ------------------------------------------------------------------
    # Data retrieval helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _track_dict(item, source_extra: Optional[dict] = None) -> Dict:
        """Build a normalised track dict from a Spotify track object."""
        d = {
            "track_id": item["id"],
            "track_name": item["name"],
            "artist_name": ", ".join(a["name"] for a in item["artists"]),
            "artist_id": item["artists"][0]["id"] if item["artists"] else "",
            "album_name": item["album"]["name"] if "album" in item else "",
            "album_image": (
                item["album"]["images"][0]["url"]
                if "album" in item and item["album"].get("images")
                else ""
            ),
            "preview_url": item.get("preview_url"),
            "duration_ms": item["duration_ms"],
            "popularity": item.get("popularity", 0),
            "explicit": item.get("explicit", False),
        }
        if source_extra:
            d.update(source_extra)
        return d

    # ------------------------------------------------------------------
    def get_recently_played(self, limit: int = 50) -> List[Dict]:
        try:
            results = self.sp.current_user_recently_played(limit=limit)
            tracks = []
            for item in results["items"]:
                t = self._track_dict(item["track"],
                                     {"played_at": item["played_at"]})
                tracks.append(t)
            logger.info(f"Retrieved {len(tracks)} recently played tracks")
            return tracks
        except Exception as e:
            logger.error(f"Error getting recently played: {e}")
            return []

    def get_top_tracks(self, time_range: str = "medium_term",
                       limit: int = 50) -> List[Dict]:
        try:
            results = self.sp.current_user_top_tracks(
                time_range=time_range, limit=limit
            )
            tracks = [self._track_dict(t) for t in results["items"]]
            logger.info(f"Retrieved {len(tracks)} top tracks ({time_range})")
            return tracks
        except Exception as e:
            logger.error(f"Error getting top tracks: {e}")
            return []

    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        try:
            all_features: List[Dict] = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i: i + 100]
                try:
                    feats = self.sp.audio_features(batch)
                    all_features.extend(f for f in feats if f is not None)
                except Exception as err:
                    logger.warning(f"Batch audio-features error: {err}")
                time.sleep(0.5)
            logger.info(f"Audio features for {len(all_features)} tracks")
            return all_features
        except Exception as e:
            logger.error(f"Error getting audio features: {e}")
            return []

    def search_tracks(self, query: str, limit: int = 20) -> List[Dict]:
        try:
            results = self.sp.search(q=query, type="track", limit=limit)
            return [self._track_dict(t) for t in results["tracks"]["items"]]
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []

    def get_user_playlists(self) -> List[Dict]:
        try:
            results = self.sp.current_user_playlists()
            return [
                {
                    "playlist_id": p["id"],
                    "playlist_name": p["name"],
                    "owner": p["owner"]["display_name"],
                    "tracks_count": p["tracks"]["total"],
                    "public": p["public"],
                }
                for p in results["items"]
            ]
        except Exception as e:
            logger.error(f"Error getting playlists: {e}")
            return []

    def get_liked_songs(self, limit: int = 50) -> List[Dict]:
        try:
            results = self.sp.current_user_saved_tracks(limit=limit)
            tracks = []
            for item in results["items"]:
                t = self._track_dict(item["track"],
                                     {"added_at": item["added_at"]})
                tracks.append(t)
            return tracks
        except Exception as e:
            logger.error(f"Error getting liked songs: {e}")
            return []

    def get_saved_albums(self, limit: int = 50) -> List[Dict]:
        try:
            results = self.sp.current_user_saved_albums(limit=limit)
            albums = []
            for item in results["items"]:
                a = item["album"]
                albums.append({
                    "album_id": a["id"],
                    "album_name": a["name"],
                    "artist_name": ", ".join(ar["name"] for ar in a["artists"]),
                    "total_tracks": a["total_tracks"],
                    "popularity": a["popularity"],
                    "added_at": item["added_at"],
                })
            return albums
        except Exception as e:
            logger.error(f"Error getting saved albums: {e}")
            return []

    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        try:
            results = self.sp.playlist_tracks(playlist_id)
            tracks = []
            for item in results["items"]:
                if item["track"] and item["track"]["type"] == "track":
                    t = self._track_dict(item["track"],
                                         {"added_at": item["added_at"]})
                    tracks.append(t)
            return tracks
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {e}")
            return []

    def get_recommendations(self, seed_tracks: List[str], limit: int = 20,
                            target_features: Optional[Dict] = None) -> List[Dict]:
        """Spotify /recommendations endpoint wrapper."""
        try:
            kwargs: dict = {"seed_tracks": seed_tracks[:5], "limit": limit}
            if target_features:
                kwargs.update(target_features)
            recs = self.sp.recommendations(**kwargs)
            tracks = [self._track_dict(t) for t in recs["tracks"]]
            logger.info(f"Got {len(tracks)} Spotify recommendations")
            return tracks
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    # ------------------------------------------------------------------
    # Comprehensive paginated fetchers
    # ------------------------------------------------------------------
    def get_user_saved_tracks_comprehensive(self, limit: int = 1000) -> List[Dict]:
        try:
            all_tracks: List[Dict] = []
            offset = 0
            while True:
                results = self.sp.current_user_saved_tracks(limit=50, offset=offset)
                if not results["items"]:
                    break
                for item in results["items"]:
                    all_tracks.append(
                        self._track_dict(item["track"],
                                         {"added_at": item["added_at"]})
                    )
                offset += 50
                if len(all_tracks) >= limit or len(results["items"]) < 50:
                    break
                time.sleep(0.1)
            logger.info(f"Comprehensive liked songs: {len(all_tracks)}")
            return all_tracks
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    def get_album_tracks(self, album_id: str) -> List[Dict]:
        try:
            results = self.sp.album_tracks(album_id)
            return [
                {
                    "track_id": t["id"],
                    "track_name": t["name"],
                    "artist_name": ", ".join(a["name"] for a in t["artists"]),
                    "album_name": "",
                    "duration_ms": t["duration_ms"],
                    "popularity": 0,
                    "explicit": t["explicit"],
                }
                for t in results["items"]
            ]
        except Exception as e:
            logger.error(f"Error getting album tracks: {e}")
            return []

    def get_user_saved_albums_comprehensive(self, limit: int = 1000) -> List[Dict]:
        try:
            all_albums: List[Dict] = []
            offset = 0
            while True:
                results = self.sp.current_user_saved_albums(limit=50, offset=offset)
                if not results["items"]:
                    break
                for item in results["items"]:
                    a = item["album"]
                    all_albums.append({
                        "album_id": a["id"],
                        "album_name": a["name"],
                        "artist_name": ", ".join(ar["name"] for ar in a["artists"]),
                        "total_tracks": a["total_tracks"],
                        "popularity": a["popularity"],
                        "added_at": item["added_at"],
                    })
                offset += 50
                if len(all_albums) >= limit or len(results["items"]) < 50:
                    break
                time.sleep(0.1)
            return all_albums
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    def get_user_playlists_comprehensive(self, limit: int = 1000) -> List[Dict]:
        try:
            all_playlists: List[Dict] = []
            offset = 0
            while True:
                results = self.sp.current_user_playlists(limit=50, offset=offset)
                if not results["items"]:
                    break
                for p in results["items"]:
                    all_playlists.append({
                        "playlist_id": p["id"],
                        "playlist_name": p["name"],
                        "owner": p["owner"]["display_name"],
                        "tracks_count": p["tracks"]["total"],
                        "public": p["public"],
                    })
                offset += 50
                if len(all_playlists) >= limit or len(results["items"]) < 50:
                    break
                time.sleep(0.1)
            return all_playlists
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    # ------------------------------------------------------------------
    # Playlist creation  (NEW – Concept 4)
    # ------------------------------------------------------------------
    def create_playlist(self, name: str, description: str = "",
                        public: bool = False) -> str:
        """Create a new playlist and return its Spotify ID."""
        user_id = self.current_user_id()
        result = self.sp.user_playlist_create(
            user=user_id, name=name, public=public, description=description
        )
        logger.info(f"Created playlist '{name}' → {result['id']}")
        return result["id"]

    def add_tracks_to_playlist(self, playlist_id: str,
                               track_ids: List[str]) -> None:
        """Add tracks to a playlist (batched in groups of 100)."""
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i: i + 100]
            uris = [f"spotify:track:{tid}" for tid in batch]
            self.sp.playlist_add_items(playlist_id, uris)
            time.sleep(0.1)
        logger.info(f"Added {len(track_ids)} tracks → playlist {playlist_id}")
