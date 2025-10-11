import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
import logging
import joblib
import os
from config import Config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing and feature engineering"""
    
    def __init__(self):
        self.config = Config()
        self.models_dir = self.config.MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize preprocessing objects
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = None
        self.kmeans = None
        
        # Feature columns for different types of analysis
        self.audio_features = [
            'danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo',
            'loudness', 'key', 'mode', 'time_signature'
        ]
        
        self.categorical_features = ['key', 'mode', 'time_signature']
        self.numerical_features = [f for f in self.audio_features if f not in self.categorical_features]
    
    def preprocess_tracks_data(self, tracks_df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess tracks data for machine learning"""
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df = tracks_df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical_features(df)
        
        # Normalize numerical features
        df = self._normalize_features(df)
        
        logger.info(f"Preprocessing completed. Final shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # For audio features, fill with median values
        for feature in self.audio_features:
            if feature in df.columns:
                if df[feature].dtype in ['int64', 'float64']:
                    df[feature] = df[feature].fillna(df[feature].median())
                else:
                    df[feature] = df[feature].fillna(df[feature].mode()[0] if not df[feature].mode().empty else 0)
        
        # For other features
        if 'popularity' in df.columns:
            df['popularity'] = df['popularity'].fillna(df['popularity'].median())
        if 'duration_ms' in df.columns:
            df['duration_ms'] = df['duration_ms'].fillna(df['duration_ms'].median())
        if 'explicit' in df.columns:
            df['explicit'] = df['explicit'].fillna(False)
        
        logger.info("Missing values handled")
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from existing ones"""
        logger.info("Engineering new features...")
        
        # Duration in minutes
        df['duration_minutes'] = df['duration_ms'] / 60000
        
        # Energy-Danceability interaction
        if 'energy' in df.columns and 'danceability' in df.columns:
            df['energy_danceability'] = df['energy'] * df['danceability']
        else:
            # Use popularity as a proxy for energy/danceability
            df['energy_danceability'] = (df['popularity'] / 100) * 0.8 + 0.1
        
        # Valence-Energy interaction (mood indicator)
        if 'valence' in df.columns and 'energy' in df.columns:
            df['mood_score'] = (df['valence'] + df['energy']) / 2
        else:
            # Use popularity as a proxy for mood
            df['mood_score'] = (df['popularity'] / 100) * 0.6 + 0.2
        
        # Acoustic vs Electronic
        if 'acousticness' in df.columns and 'energy' in df.columns:
            df['acoustic_electronic'] = df['acousticness'] - df['energy']
        else:
            # Use popularity as a proxy (higher popularity = more electronic)
            df['acoustic_electronic'] = -((df['popularity'] / 100) * 0.4 - 0.2)
        
        # Complexity score (combination of multiple features)
        complexity_features = [f for f in ['speechiness', 'instrumentalness', 'liveness'] if f in df.columns]
        if complexity_features:
            df['complexity_score'] = df[complexity_features].mean(axis=1)
        else:
            # Use duration as a proxy for complexity
            df['complexity_score'] = np.clip((df['duration_ms'] / 300000) * 0.3 + 0.2, 0, 1)
        
        # Tempo categories
        if 'tempo' in df.columns:
            df['tempo_category'] = pd.cut(
                df['tempo'], 
                bins=[0, 80, 120, 160, 200, 300], 
                labels=['very_slow', 'slow', 'medium', 'fast', 'very_fast']
            )
        else:
            df['tempo_category'] = 'medium'
        
        # Energy categories
        if 'energy' in df.columns:
            df['energy_category'] = pd.cut(
                df['energy'], 
                bins=[0, 0.3, 0.6, 0.8, 1.0], 
                labels=['low', 'medium', 'high', 'very_high']
            )
        else:
            df['energy_category'] = 'medium'
        
        # Valence categories (mood)
        if 'valence' in df.columns:
            df['valence_category'] = pd.cut(
                df['valence'], 
                bins=[0, 0.3, 0.6, 1.0], 
                labels=['sad', 'neutral', 'happy']
            )
        else:
            df['valence_category'] = 'neutral'
        
        # Artist diversity (how many different artists in user's top tracks)
        if 'artist_name' in df.columns:
            artist_counts = df['artist_name'].value_counts()
            df['artist_diversity_score'] = df['artist_name'].map(artist_counts)
            df['artist_diversity_score'] = 1 / df['artist_diversity_score']  # Inverse for diversity
        
        logger.info("Feature engineering completed")
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = ['key', 'mode', 'time_signature', 'tempo_category', 
                           'energy_category', 'valence_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories
                    unique_values = df[col].astype(str).unique()
                    known_values = self.label_encoders[col].classes_
                    unseen_values = set(unique_values) - set(known_values)
                    
                    if unseen_values:
                        # Add unseen values to encoder
                        all_values = list(known_values) + list(unseen_values)
                        self.label_encoders[col] = LabelEncoder()
                        self.label_encoders[col].fit(all_values)
                    
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        logger.info("Categorical encoding completed")
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features"""
        logger.info("Normalizing features...")
        
        # Features to normalize
        features_to_normalize = self.numerical_features + [
            'duration_minutes', 'energy_danceability', 'mood_score',
            'acoustic_electronic', 'complexity_score', 'artist_diversity_score'
        ]
        
        # Filter features that exist in the dataframe
        existing_features = [f for f in features_to_normalize if f in df.columns]
        
        if existing_features:
            df[existing_features] = self.scaler.fit_transform(df[existing_features])
        
        logger.info("Feature normalization completed")
        return df
    
    def create_user_profile_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Create a comprehensive user profile vector from their listening history"""
        logger.info("Creating comprehensive user profile vector...")
        
        # Weight tracks by their importance (top tracks have higher weight)
        weights = df['weight'].values if 'weight' in df.columns else np.ones(len(df))
        
        # Calculate weighted averages for audio features
        profile_vector = []
        
        # Core audio features
        for feature in self.audio_features:
            if feature in df.columns:
                weighted_avg = np.average(df[feature].values, weights=weights)
                profile_vector.append(weighted_avg)
        
        # Add engineered features
        engineered_features = [
            'duration_minutes', 'energy_danceability', 'mood_score',
            'acoustic_electronic', 'complexity_score'
        ]
        
        for feature in engineered_features:
            if feature in df.columns:
                weighted_avg = np.average(df[feature].values, weights=weights)
                profile_vector.append(weighted_avg)
        
        # Add listening pattern features
        if 'popularity' in df.columns:
            weighted_avg = np.average(df['popularity'].values, weights=weights)
            profile_vector.append(weighted_avg / 100)  # Normalize to 0-1
        
        # Add diversity metrics
        if 'artist_name' in df.columns:
            unique_artists = df['artist_name'].nunique()
            total_tracks = len(df)
            diversity_score = unique_artists / total_tracks if total_tracks > 0 else 0
            profile_vector.append(diversity_score)
        
        # Add tempo preference (normalized)
        if 'tempo' in df.columns:
            tempo_avg = np.average(df['tempo'].values, weights=weights)
            # Normalize tempo to 0-1 range (assuming 60-200 BPM range)
            normalized_tempo = (tempo_avg - 60) / 140 if tempo_avg > 60 else 0
            profile_vector.append(min(normalized_tempo, 1.0))
        
        # Add energy-danceability interaction
        if 'energy' in df.columns and 'danceability' in df.columns:
            energy_avg = np.average(df['energy'].values, weights=weights)
            dance_avg = np.average(df['danceability'].values, weights=weights)
            interaction = energy_avg * dance_avg
            profile_vector.append(interaction)
        
        # Add valence-energy interaction (mood-energy)
        if 'valence' in df.columns and 'energy' in df.columns:
            valence_avg = np.average(df['valence'].values, weights=weights)
            energy_avg = np.average(df['energy'].values, weights=weights)
            mood_energy = valence_avg * energy_avg
            profile_vector.append(mood_energy)
        
        profile_vector = np.array(profile_vector)
        logger.info(f"Comprehensive user profile vector created with {len(profile_vector)} features")
        
        return profile_vector
    
    def perform_pca(self, df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
        """Perform PCA dimensionality reduction"""
        logger.info(f"Performing PCA with {n_components} components...")
        
        # Select features for PCA
        pca_features = [f for f in self.audio_features if f in df.columns]
        pca_features.extend([
            'duration_minutes', 'energy_danceability', 'mood_score',
            'acoustic_electronic', 'complexity_score'
        ])
        
        # Filter existing features
        pca_features = [f for f in pca_features if f in df.columns]
        
        if len(pca_features) < n_components:
            n_components = len(pca_features)
        
        # Perform PCA
        self.pca = PCA(n_components=n_components)
        pca_data = self.pca.fit_transform(df[pca_features])
        
        # Create new columns for PCA components
        pca_columns = [f'pca_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_data, columns=pca_columns, index=df.index)
        
        # Combine with original data
        result_df = pd.concat([df, pca_df], axis=1)
        
        logger.info(f"PCA completed. Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        return result_df
    
    def cluster_tracks(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """Cluster tracks based on audio features"""
        logger.info(f"Clustering tracks into {n_clusters} clusters...")
        
        # Use PCA components if available, otherwise use original features
        cluster_features = [f for f in df.columns if f.startswith('pca_')]
        
        if not cluster_features:
            cluster_features = [f for f in self.audio_features if f in df.columns]
        
        if not cluster_features:
            logger.warning("No suitable features for clustering found")
            return df
        
        # Perform clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(df[cluster_features])
        
        # Add cluster labels to dataframe
        df['cluster'] = cluster_labels
        
        logger.info("Track clustering completed")
        return df
    
    def save_preprocessing_objects(self):
        """Save preprocessing objects for later use"""
        logger.info("Saving preprocessing objects...")
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, os.path.join(self.models_dir, f'label_encoder_{name}.pkl'))
        
        # Save PCA
        if self.pca is not None:
            joblib.dump(self.pca, os.path.join(self.models_dir, 'pca.pkl'))
        
        # Save KMeans
        if self.kmeans is not None:
            joblib.dump(self.kmeans, os.path.join(self.models_dir, 'kmeans.pkl'))
        
        logger.info("Preprocessing objects saved")
    
    def load_preprocessing_objects(self):
        """Load preprocessing objects"""
        logger.info("Loading preprocessing objects...")
        
        try:
            # Load scaler
            scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load label encoders
            for file in os.listdir(self.models_dir):
                if file.startswith('label_encoder_'):
                    name = file.replace('label_encoder_', '').replace('.pkl', '')
                    self.label_encoders[name] = joblib.load(os.path.join(self.models_dir, file))
            
            # Load PCA
            pca_path = os.path.join(self.models_dir, 'pca.pkl')
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
            
            # Load KMeans
            kmeans_path = os.path.join(self.models_dir, 'kmeans.pkl')
            if os.path.exists(kmeans_path):
                self.kmeans = joblib.load(kmeans_path)
            
            logger.info("Preprocessing objects loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading preprocessing objects: {str(e)}")
            raise
