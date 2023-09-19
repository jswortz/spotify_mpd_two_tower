import json
import numpy as np
import pickle as pkl
import os
from pprint import pprint
from typing import Dict, Text, List, Optional

import tensorflow as tf
import tensorflow_recommenders as tfrs

from google.cloud import storage

# ================================================================
# TODO - handle relative imports for local and cloud execution
# ================================================================

# # relative imports running locally
from src.ranking import train_utils #, feature_sets
from src.ranking import train_config as cfg

# # relative imports running cloud
# # import feature_sets
# import train_utils
# import train_config as cfg

# ================================================================

MAX_PLAYLIST_LENGTH = cfg.TRACK_HISTORY # 5 | cfg.MAX_PLAYLIST_LENGTH
# PROJECT_ID = cfg.PROJECT_ID # 'hybrid-vertex' | cfg.PROJECT_ID
# project_number = os.environ["CLOUD_ML_PROJECT_ID"]

# ========================================
# ranking baseline model
# ========================================
class RankingModel(tf.keras.Model):
    '''
    build sequential model for each feature
    pass outputs to dense/cross layers
    concatentate the outputs
    '''
    def __init__(
        self
        , vocab_dict
        , layer_sizes: List = [512, 256, 128]
        , embedding_dim: int = 128
        , projection_dim: int = 50
        , seed: int = 1234
        , use_dropout: bool = True
        , dropout_rate: float = 0.33
        , max_tokens: int = 20000
    ):
        super().__init__()
        
        # ========================================
        # non-sequence playlist features
        # ========================================
        
        # Feature: pl_name_src
        self.pl_name_src_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['pl_name_src'],
                    ngrams=2, 
                    name="pl_name_src_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="pl_name_src_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="pl_name_src_1d"),
            ], name="pl_name_src_text_embedding"
        )
        
        # Feature: pl_collaborative_src
        self.pl_collaborative_src_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=3), 
                tf.keras.layers.Embedding(
                    input_dim=3 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="pl_collaborative_emb_layer",
                    input_shape=()
                ),
            ], name="pl_collaborative_emb_model"
        )
        
        # Feature: pl_duration_ms_new
        self.pl_duration_ms_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(635073792)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="pl_duration_ms_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="pl_duration_ms_new_1d"),
            ], name="pl_duration_ms_new_emb_model"
        )
        
        # Feature: num_pl_songs_new | n_songs_pl_new
        self.num_pl_songs_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(376)),    # TODO - 376 from TRAIN
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="num_pl_songs_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_songs_new_1d"),
            ], name="num_pl_songs_new_emb_model"
        )
        
        # Feature: num_pl_artists_new
        self.num_pl_artists_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(238)),     # TODO - 238 from TRAIN
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="num_pl_artists_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_artists_new_1d"),
            ], name="num_pl_artists_new_emb_model"
        )
        
        # Feature: num_pl_albums_new
        self.num_pl_albums_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(244)),   # TODO - 244 from TRAIN
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="num_pl_albums_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_albums_new_1d"),  
            ], name="num_pl_albums_new_emb_model"
        )
        
        # ========================================
        # sequence playlist features
        # ========================================
        
        # Feature: track_uri_pl
        # 2.2M unique
        self.track_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2249561, mask_value=''), # 2249561
                tf.keras.layers.Embedding(
                    input_dim=2249561 + 1, 
                    output_dim=embedding_dim,
                    name="track_uri_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_uri_1d"),
            ], name="track_uri_pl_emb_model"
        )
        
        # Feature: track_name_pl
        self.track_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens,
                    ngrams=2, 
                    vocabulary=vocab_dict['track_name_pl'],
                    # vocabulary=np.array([vocab_dict['track_name_pl']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=MAX_PLAYLIST_LENGTH,
                    name="track_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, # + 1, 
                    output_dim=embedding_dim,
                    name="track_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="track_name_pl_2d"),
            ], name="track_name_pl_emb_model"
        )
        
        # Feature: artist_uri_pl
        self.artist_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=295860, mask_value=''), # 295860
                tf.keras.layers.Embedding(
                    input_dim=295860 + 1, 
                    output_dim=embedding_dim,
                    name="artist_uri_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_uri_pl_1d"),
            ], name="artist_uri_pl_emb_model"
        )
        
        # Feature: artist_name_pl
        self.artist_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    ngrams=2, 
                    vocabulary=vocab_dict['artist_name_pl'],
                    name="artist_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, #  + 1, 
                    output_dim=embedding_dim,
                    name="artist_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_name_pl_2d"),
            ], name="artist_name_pl_emb_model"
        )
        
        # Feature: album_uri_pl
        self.album_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=730377, mask_value=''), # 730377
                tf.keras.layers.Embedding(
                    input_dim=730377 + 1, 
                    output_dim=embedding_dim,
                    name="album_uri_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_uri_pl_1d"),
            ], name="album_uri_pl_emb_model"
        )
        
        # Feature: album_name_pl
        self.album_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    ngrams=2, 
                    vocabulary=vocab_dict['album_name_pl'],
                    name="album_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, #  + 1, # 571625 + 1, 
                    output_dim=embedding_dim,
                    name="album_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="album_name_pl_emb_layer_2d"),
            ], name="album_name_pl_emb_model"
        )
        
        # Feature: artist_genres_pl
        self.artist_genres_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    ngrams=2, 
                    vocabulary=vocab_dict['artist_genres_pl'],
                    name="artist_genres_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens, #  + 1, 
                    output_dim=embedding_dim,
                    name="artist_genres_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, embedding_dim]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_genres_pl_2d"),
            ], name="artist_genres_pl_emb_model"
        )
        
        # Feature: duration_ms_songs_pl
        self.duration_ms_songs_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(20744575)), # 20744575.0
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="duration_ms_songs_pl_emb_layer",
                    mask_zero=False
                ),
            tf.keras.layers.GlobalAveragePooling1D(name="duration_ms_songs_pl_emb_layer_pl_1d"),
            ], name="duration_ms_songs_pl_emb_model"
        )
        
        # Feature: track_pop_pl
        self.track_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_pop_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_pop_pl_1d"),
            ], name="track_pop_pl_emb_model"
        )
        
        # Feature: artist_pop_pl
        self.artist_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="artist_pop_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_pop_1d"),
            ], name="artist_pop_pl_emb_model"
        )
        
        # Feature: artists_followers_pl
        self.artists_followers_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(94437255)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="artists_followers_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artists_followers_pl_1d"),
            ], name="artists_followers_pl_emb_model"
        )
        
        # Feature: track_danceability_pl
        self.track_danceability_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_danceability_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_danceability_pl_1d"),
            ], name="track_danceability_pl_emb_model"
        )
        
        # Feature: track_energy_pl
        self.track_energy_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_energy_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_energy_pl_1d"),
            ], name="track_energy_pl_emb_model"
        )
        
        # Feature: track_key_pl
        self.track_key_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=12), # , mask_value=''), # 2249561
                tf.keras.layers.Embedding(
                    input_dim=12 + 1, 
                    output_dim=embedding_dim,
                    name="track_key_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_key_pl_1d"),
            ], name="track_key_pl_emb_model"
        )
        
        # Feature: track_loudness_pl
        self.track_loudness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(5)), # TODO - Normalize? [-60, 5)
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_loudness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_loudness_pl_1d"),
            ], name="track_loudness_pl_emb_model"
        )
        
        # Feature: track_mode_pl
        self.track_mode_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=3),
                tf.keras.layers.Embedding(
                    input_dim=3 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="track_mode_pl_emb_layer",
                    input_shape=()
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_mode_pl_1d"),
            ], name="track_mode_pl_emb_model"
        )
        
        # Feature: track_speechiness_pl
        self.track_speechiness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_speechiness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_speechiness_pl_1d"),
            ], name="track_speechiness_pl_emb_model"
        )
        
        # Feature: track_acousticness_pl
        self.track_acousticness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_acousticness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_acousticness_pl_1d"),
            ], name="track_acousticness_pl_emb_model"
        )
        
        # Feature: track_instrumentalness_pl
        self.track_instrumentalness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_instrumentalness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_instrumentalness_pl_1d"),
            ], name="track_instrumentalness_pl_emb_model"
        )
        
        # Feature: track_liveness_pl
        self.track_liveness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_liveness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_liveness_pl_1d"),
            ], name="track_liveness_pl_emb_model"
        )
        
        # Feature: track_valence_pl
        self.track_valence_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_valence_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_valence_pl_1d"),
            ], name="track_valence_pl_emb_model"
        )
        
        # Feature: track_tempo_pl
        self.track_tempo_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(250)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_tempo_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_tempo_pl_1d"),
            ], name="track_tempo_pl_emb_model"
        )
        
        # Feature: time_signature_pl
        self.time_signature_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=6),
                tf.keras.layers.Embedding(
                    input_dim=6 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="time_signature_pl_emb_layer",
                    input_shape=()
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="time_signature_pl_1d"),
            ], name="time_signature_pl_emb_model"
        )
        
        # ========================================
        # Candidate features
        # ========================================
        
        # Feature: track_uri_can
        self.track_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2249561), # 2249561
                tf.keras.layers.Embedding(
                    input_dim=2249561+1, 
                    output_dim=embedding_dim,
                    name="track_uri_can_emb_layer",
                ),
            ], name="track_uri_can_emb_model"
        )
        
        # Feature: track_name_can
        self.track_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['track_name_can'],
                    ngrams=2, 
                    name="track_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens+1,
                    output_dim=embedding_dim,
                    name="track_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_name_can_1d"),
            ], name="track_name_can_emb_model"
        )
        
        # Feature: artist_uri_can
        self.artist_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=295860), # 295860
                tf.keras.layers.Embedding(
                    input_dim=295860+1, 
                    output_dim=embedding_dim,
                    name="artist_uri_can_emb_layer",
                ),
            ], name="artist_uri_can_emb_model"
        )
        
        # Feature: artist_name_can
        self.artist_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['artist_name_can'],
                    ngrams=2, 
                    name="artist_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens+1,
                    output_dim=embedding_dim,
                    name="artist_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_name_can_1d"),
            ], name="artist_name_can_emb_model"
        )
        
        # Feature: album_uri_can
        self.album_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=734684), # 734684
                tf.keras.layers.Embedding(
                    input_dim=734684+1, 
                    output_dim=embedding_dim,
                    name="album_uri_can_emb_layer",
                ),
            ], name="album_uri_can_emb_model"
        )

        # Feature: album_name_can
        self.album_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['album_name_can'],
                    ngrams=2, 
                    name="album_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens+1,
                    output_dim=embedding_dim,
                    name="album_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_name_can_1d"),
            ], name="album_name_can_emb_model"
        )
        
        # Feature: duration_ms_can
        self.duration_ms_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(20744575)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="duration_ms_can_emb_layer",
                ),
            ], name="duration_ms_can_emb_model"
        )
        
        # Feature: track_pop_can
        self.track_pop_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_pop_can_emb_layer",
                ),
            ], name="track_pop_can_emb_model"
        )
        
        # Feature: artist_pop_can
        self.artist_pop_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="artist_pop_can_emb_layer",
                ),
            ], name="artist_pop_can_emb_model"
        )
        
        # Feature: artist_genres_can
        self.artist_genres_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=max_tokens, 
                    vocabulary=vocab_dict['artist_genres_can'],
                    ngrams=2, 
                    name="artist_genres_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=max_tokens + 1, 
                    output_dim=embedding_dim,
                    name="artist_genres_can_emb_layer",
                    mask_zero=True
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_can_1d"),
            ], name="artist_genres_can_emb_model"
        )
        
        # Feature: artist_followers_can
        self.artists_followers_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(94437255)), # TODO - was 94437255
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="artists_followers_can_emb_layer",
                ),
            ], name="artists_followers_can_emb_model"
        )
        
        # track_danceability_can
        self.track_danceability_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_danceability_can_emb_layer",
                ),
            ], name="track_danceability_can_emb_model"
        )
        
        # track_energy_can
        self.track_energy_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_energy_can_emb_layer",
                ),
            ], name="track_energy_can_emb_model"
        )
        
        # track_key_can
        self.track_key_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=12), #, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=12 + 1, 
                    output_dim=embedding_dim,
                    name="track_key_can_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_key_can_1d"),
            ], name="track_key_can_emb_model"
        )
        
        # track_loudness_can
        self.track_loudness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(5)), # TODO - Normalize? [-60, 5)
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_loudness_can_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_loudness_can_1d"),
            ], name="track_loudness_can_emb_model"
        )
        
        # track_mode_can
        self.track_mode_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=3),
                tf.keras.layers.Embedding(
                    input_dim=3 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="track_mode_can_emb_layer",
                    input_shape=()
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_mode_can_1d"),
            ], name="track_mode_can_emb_model"
        )
        
        # track_speechiness_can
        self.track_speechiness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_speechiness_can_emb_layer",
                ),
            ], name="track_speechiness_can_emb_model"
        )
        
        # track_acousticness_can
        self.track_acousticness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_acousticness_can_emb_layer",
                ),
            ], name="track_acousticness_can_emb_model"
        )
        
        # track_instrumentalness_can
        self.track_instrumentalness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_instrumentalness_can_emb_layer",
                ),
            ], name="track_instrumentalness_can_emb_model"
        )
        
        # track_liveness_can
        self.track_liveness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_liveness_can_emb_layer",
                ),
            ], name="track_liveness_can_emb_model"
        )
        
        # track_valence_can
        self.track_valence_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_valence_can_emb_layer",
                ),
            ], name="track_valence_can_emb_model"
        )
        
        # track_tempo_can
        self.track_tempo_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(train_utils.get_buckets_20(250)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=embedding_dim,
                    name="track_tempo_can_emb_layer",
                ),
            ], name="track_tempo_can_emb_model"
        )
        
        # time_signature_can
        self.time_signature_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=6),
                tf.keras.layers.Embedding(
                    input_dim=6 + 1,
                    output_dim=embedding_dim,
                    mask_zero=False,
                    name="time_signature_can_emb_layer",
                    input_shape=()
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="time_signature_can_1d"),
            ], name="time_signature_can_emb_model"
        )
        
        
        # ========================================
        # Ranking - dense layers
        # ========================================

        # Dense Layers
        self.rank_dense_layers = tf.keras.Sequential(name="rank_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.rank_dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
                )
            )
            if use_dropout:
                self.rank_dense_layers.add(tf.keras.layers.Dropout(dropout_rate))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.rank_dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
                )
            )
        
        # L2 Norm
        self.rank_dense_layers.add(tf.keras.layers.LayerNormalization(name="normalize_dense"))
        
        # Make rank predictions in final layer
        self.rank_dense_layers.add(tf.keras.layers.Dense(1, name="final_layer"))
        
      #   # Compute predictions.
      #   self.ratings = tf.keras.Sequential([
      #     # Learn multiple dense layers.
      #     tf.keras.layers.Dense(256, activation="relu"),
      #     tf.keras.layers.Dense(64, activation="relu"),
      #     # Make rating predictions in the final layer.
      #     tf.keras.layers.Dense(1)
      # ])
        
    # ========================================
    # call
    # ========================================
    def call(self, data):
        '''
        The call method defines what happens when
        the model is called
        '''
       
        all_embs = tf.concat(
            [
                self.pl_name_src_text_embedding(data['pl_name_src']),
                self.pl_collaborative_src_embedding(data['pl_collaborative_src']),
                self.pl_duration_ms_new_embedding(data["pl_duration_ms_new"]),
                self.num_pl_songs_new_embedding(data["num_pl_songs_new"]),
                self.num_pl_artists_new_embedding(data["num_pl_artists_new"]),
                self.num_pl_albums_new_embedding(data["num_pl_albums_new"]),
                
                # sequence features
                self.track_uri_pl_embedding(data['track_uri_pl']),
                self.track_name_pl_embedding(tf.reshape(data['track_name_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.artist_uri_pl_embedding(data["artist_uri_pl"]),
                self.artist_name_pl_embedding(tf.reshape(data["artist_name_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.album_uri_pl_embedding(data["album_uri_pl"]),
                self.album_name_pl_embedding(tf.reshape(data["album_name_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.artist_genres_pl_embedding(tf.reshape(data["artist_genres_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                
                self.duration_ms_songs_pl_embedding(data["duration_ms_songs_pl"]),
                self.track_pop_pl_embedding(data["track_pop_pl"]),
                self.artist_pop_pl_embedding(data["artist_pop_pl"]),
                self.artists_followers_pl_embedding(data["artists_followers_pl"]),
                self.track_danceability_pl_embedding(data["track_danceability_pl"]),
                self.track_energy_pl_embedding(data["track_energy_pl"]),
                self.track_key_pl_embedding(data["track_key_pl"]),
                self.track_loudness_pl_embedding(data["track_loudness_pl"]),
                self.track_mode_pl_embedding(data["track_mode_pl"]),
                self.track_speechiness_pl_embedding(data["track_speechiness_pl"]),
                self.track_acousticness_pl_embedding(data["track_acousticness_pl"]),
                self.track_instrumentalness_pl_embedding(data["track_instrumentalness_pl"]),
                self.track_liveness_pl_embedding(data["track_liveness_pl"]),
                self.track_valence_pl_embedding(data["track_valence_pl"]),
                self.track_tempo_pl_embedding(data["track_tempo_pl"]),
                self.time_signature_pl_embedding(data["track_time_signature_pl"]),
                
                # candidate features
                self.track_uri_can_embedding(data['track_uri_can']),
                self.track_name_can_embedding(data['track_name_can']),
                self.artist_uri_can_embedding(data['artist_uri_can']),
                self.artist_name_can_embedding(data['artist_name_can']),
                self.album_uri_can_embedding(data['album_uri_can']),
                self.album_name_can_embedding(data['album_name_can']),
                self.duration_ms_can_embedding(data['duration_ms_can']),
                self.track_pop_can_embedding(data['track_pop_can']),
                self.artist_pop_can_embedding(data['artist_pop_can']),
                self.artist_genres_can_embedding(data['artist_genres_can']),
                self.artists_followers_can_embedding(data['artist_followers_can']),

                self.track_danceability_can_embedding(data['track_danceability_can']),
                self.track_energy_can_embedding(data['track_energy_can']),
                self.track_key_can_embedding(data['track_key_can']),
                self.track_loudness_can_embedding(data['track_loudness_can']),
                self.track_mode_can_embedding(data['track_mode_can']),
                self.track_speechiness_can_embedding(data['track_speechiness_can']),
                self.track_acousticness_can_embedding(data['track_acousticness_can']),
                self.track_instrumentalness_can_embedding(data['track_instrumentalness_can']),
                self.track_liveness_can_embedding(data['track_liveness_can']),
                self.track_valence_can_embedding(data['track_valence_can']),
                self.track_tempo_can_embedding(data['track_tempo_can']),
                self.time_signature_can_embedding(data['track_time_signature_can']),
                
            ], axis=1)
        
        return self.rank_dense_layers(all_embs)
    
# ========================================
# Ranking - compute predictions
# ========================================

class TheRankingModel(tfrs.models.Model):

    # here

    def __init__(
        self
        , vocab_dict
        , layer_sizes: List = [512, 256, 128]
        , embedding_dim: int = 128
        , projection_dim: int = 50
        , seed: int = 1234
        , use_dropout: bool = True
        , dropout_rate: float = 0.33
        , max_tokens: int = 20000
    ):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(
            seed = seed
            , vocab_dict = vocab_dict
            , max_tokens = max_tokens
            , use_dropout = use_dropout
            , layer_sizes = layer_sizes
            , dropout_rate = dropout_rate
            , embedding_dim = embedding_dim
            , projection_dim = projection_dim
        )

        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(
                # reduction=losses_utils.ReductionV2.AUTO
                reduction=tf.keras.losses.Reduction.SUM
            )
            , metrics = [tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model(features)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        
        labels = features.pop("candidate_rank")

        rating_predictions = self(features)

        # The task computes the loss and the metrics.
        return self.task(
            labels=labels
            , predictions=rating_predictions
        )