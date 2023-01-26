import json
import tensorflow as tf
import tensorflow_recommenders as tfrs

from google.cloud import storage


import numpy as np
import pickle as pkl
import os
from pprint import pprint

# TODO: formatting for below import 
import train_config as cfg # needed for Vertex Train package
# from . import train_config as cfg # needed for `02-build-model.ipynb`

EMBEDDING_DIM = cfg.EMBEDDING_DIM       # 128
PROJECTION_DIM = cfg.PROJECTION_DIM     # 50
SEED = cfg.SEED                         # 1234
USE_CROSS_LAYER = cfg.USE_CROSS_LAYER   # True
DROPOUT = cfg.USE_DROPOUT               # 'False'
DROPOUT_RATE = cfg.DROPOUT_RATE         # '0.33'
MAX_PLAYLIST_LENGTH = cfg.MAX_PLAYLIST_LENGTH   # 5
MAX_TOKENS = cfg.MAX_TOKENS
PROJECT_ID = cfg.PROJECT_ID # '20000'

project_number = os.environ["CLOUD_ML_PROJECT_ID"]

storage_client = storage.Client(project=project_number)

# ====================================================
# helper functions
# ====================================================

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name" (no 'gs://')
    # source_file_name = "local/path/to/file" (file to upload)
    # destination_blob_name = "folder/paths-to/storage-object-name"
    storage_client = storage.Client(project=project_number)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
    
def get_buckets_20(MAX_VAL):
    """ 
    creates discretization buckets of size 20
    """
    list_buckets = list(np.linspace(0, MAX_VAL, num=20))
    return(list_buckets)

candidate_features = {
    "track_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),            
    "track_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "album_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),           
    "album_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
    "duration_ms_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
    "track_pop_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
    "artist_pop_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "artist_genres_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_followers_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    # new
    # "track_pl_titles_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_danceability_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_energy_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_key_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_loudness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_mode_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_speechiness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_acousticness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_instrumentalness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_liveness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_valence_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_tempo_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "time_signature_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
}

feats = {
    # ===================================================
    # candidate track features
    # ===================================================
    "track_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),            
    "track_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "album_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),           
    "album_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
    "duration_ms_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
    "track_pop_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
    "artist_pop_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "artist_genres_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_followers_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    # "track_pl_titles_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_danceability_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_energy_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_key_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_loudness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_mode_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_speechiness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_acousticness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_instrumentalness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_liveness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_valence_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_tempo_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "time_signature_can": tf.io.FixedLenFeature(dtype=tf.string, shape=()), # track_time_signature_can
    
    # ===================================================
    # summary playlist features
    # ===================================================
    "pl_name_src" : tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
    'pl_collaborative_src' : tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
    # 'num_pl_followers_src' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
    'pl_duration_ms_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_pl_songs_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), # n_songs_pl_new | num_pl_songs_new
    'num_pl_artists_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_pl_albums_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
    # 'avg_track_pop_pl_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
    # 'avg_artist_pop_pl_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
    # 'avg_art_followers_pl_new' : tf.io.FixedLenFeature(dtype=tf.float32, shape=()), 
    
    # ===================================================
    # ragged playlist features
    # ===================================================
    # bytes / string
    "track_uri_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "artist_uri_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "artist_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "album_uri_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "album_name_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "artist_genres_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    # "tracks_playlist_titles_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_key_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_mode_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    "time_signature_pl": tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)), 

    # Float List
    "duration_ms_songs_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_pop_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "artist_pop_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "artists_followers_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_danceability_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_energy_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_loudness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_speechiness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_acousticness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_instrumentalness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_liveness_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_valence_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    "track_tempo_pl": tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
}

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA


def parse_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example,
        feats
        # features=feats
    )
    return example


def parse_candidate_tfrecord_fn(example):
    """
    Reads candidate serialized examples from gcs and converts to tfrecord
    """
    # example = tf.io.parse_single_example(
    example = tf.io.parse_example(
        example, 
        features=candidate_features
    )
    return example

# ======================
# Vocab Adapts
# ======================
# > TODO: vocab adapts, min/max, etc. - from train set only?
# > TODO: think about Feature Store integration

import pickle as pkl

os.system('gsutil cp gs://two-tower-models/vocabs/vocab_dict.pkl .')  # TODO - paramterize

filehandler = open('vocab_dict.pkl', 'rb')
vocab_dict = pkl.load(filehandler)
filehandler.close()

# ========================================
# playlist tower
# ========================================
class Playlist_Model(tf.keras.Model):
    '''
    build sequential model for each feature
    pass outputs to dense/cross layers
    concatentate the outputs
    
    the produced embedding represents the features 
    of a Playlist known at query time 
    '''
    def __init__(self, layer_sizes, vocab_dict):   # vocab_dict
        super().__init__()
        
        # ========================================
        # non-sequence playlist features
        # ========================================
        
        # Feature: pl_name_src
        self.pl_name_src_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    vocabulary=vocab_dict['pl_name_src'],
                    ngrams=2, 
                    name="pl_name_src_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS, # + 1, 
                    output_dim=EMBEDDING_DIM,
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
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="pl_collaborative_emb_layer",
                    input_shape=()
                ),
            ], name="pl_collaborative_emb_model"
        )
        
        # # Feature: num_pl_followers_src
        # self.num_pl_followers_src_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Discretization(get_buckets_20(71643)),
        #         tf.keras.layers.Embedding(
        #             input_dim=20 + 1, 
        #             output_dim=EMBEDDING_DIM,
        #             name="num_pl_followers_src_emb_layer",
        #         ),
        #         # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_followers_src_1d"),
        #     ], name="num_pl_followers_src_emb_model"
        # )
        
        # Feature: pl_duration_ms_new
        self.pl_duration_ms_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(635073792)), 
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="pl_duration_ms_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="pl_duration_ms_new_1d"),
            ], name="pl_duration_ms_new_emb_model"
        )
        
        # Feature: num_pl_songs_new | n_songs_pl_new
        self.num_pl_songs_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(376)),    # TODO - 376 from TRAIN
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="num_pl_songs_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_songs_new_1d"),
            ], name="num_pl_songs_new_emb_model"
        )
        
        # Feature: num_pl_artists_new
        self.num_pl_artists_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(238)),     # TODO - 238 from TRAIN
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="num_pl_artists_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_artists_new_1d"),
            ], name="num_pl_artists_new_emb_model"
        )
        
        # Feature: num_pl_albums_new
        self.num_pl_albums_new_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(244)),   # TODO - 244 from TRAIN
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="num_pl_albums_new_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="num_pl_albums_new_1d"),  
            ], name="num_pl_albums_new_emb_model"
        )
        
#         # Feature: avg_track_pop_pl_new
#         self.avg_track_pop_pl_new_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Discretization(get_buckets_20(86)),      # TODO 86 from TRAIN
#                 tf.keras.layers.Embedding(
#                     input_dim=20 + 1, 
#                     output_dim=EMBEDDING_DIM,
#                     name="avg_track_pop_pl_new_emb_layer",
#                 ),
#                 # tf.keras.layers.GlobalAveragePooling1D(name="avg_track_pop_pl_new_1d"),   
#             ], name="avg_track_pop_pl_new_emb_model"
#         )
        
#         # Feature: avg_artist_pop_pl_new
#         self.avg_artist_pop_pl_new_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Discretization(get_buckets_20(100)),     # TODO - parametrize
#                 tf.keras.layers.Embedding(
#                     input_dim=20 + 1, 
#                     output_dim=EMBEDDING_DIM,
#                     name="avg_artist_pop_pl_new_emb_layer",
#                 ),
#                 # tf.keras.layers.GlobalAveragePooling1D(name="avg_artist_pop_pl_new_1d"),
#             ], name="avg_artist_pop_pl_new_emb_model"
#         )
        
#         # Feature: avg_art_followers_pl_new
#         self.avg_art_followers_pl_new_embedding = tf.keras.Sequential(
#             [
#                 tf.keras.layers.Discretization(get_buckets_20(93079438)),   # TODO - 93079438 from TRAIN
#                 tf.keras.layers.Embedding(
#                     input_dim=20 + 1, 
#                     output_dim=EMBEDDING_DIM,
#                     name="avg_art_followers_pl_new_emb_layer",
#                 ),
#                 # tf.keras.layers.GlobalAveragePooling1D(name="avg_art_followers_pl_new_1d"),
#             ], name="avg_art_followers_pl_new_emb_model"
#         )
        
        # ========================================
        # sequence playlist features
        # ========================================
        
        # Feature: track_uri_pl
        # 2.2M unique
        self.track_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2249561, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=2249561 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_uri_1d"),
            ], name="track_uri_pl_emb_model"
        )
        
        # Feature: track_name_pl
        self.track_name_pl_embedding = tf.keras.Sequential(
            [
                # tf.keras.layers.Reshape((-1, MAX_PLAYLIST_LENGTH, 1), input_shape=(MAX_PLAYLIST_LENGTH,)),
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS,
                    ngrams=2, 
                    vocabulary=vocab_dict['track_name_pl'],
                    # vocabulary=np.array([vocab_dict['track_name_pl']]).flatten(),
                    # output_mode='int',
                    # output_sequence_length=MAX_PLAYLIST_LENGTH,
                    name="track_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS, # + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="track_name_pl_2d"),
            ], name="track_name_pl_emb_model"
        )
        
        # Feature: artist_uri_pl
        self.artist_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=295860, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=295860 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_uri_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_uri_pl_1d"),
            ], name="artist_uri_pl_emb_model"
        )
        
        # Feature: artist_name_pl
        self.artist_name_pl_embedding = tf.keras.Sequential(
            [
                # tf.keras.layers.Reshape((-1, MAX_PLAYLIST_LENGTH, 1), input_shape=(MAX_PLAYLIST_LENGTH,)),
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    ngrams=2, 
                    vocabulary=vocab_dict['artist_name_pl'],
                    name="artist_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS, #  + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_name_pl_2d"),
            ], name="artist_name_pl_emb_model"
        )
        
        # Feature: album_uri_pl
        self.album_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=730377, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=730377 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="album_uri_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_uri_pl_1d"),
            ], name="album_uri_pl_emb_model"
        )
        
        # Feature: album_name_pl
        self.album_name_pl_embedding = tf.keras.Sequential(
            [
                # tf.keras.layers.Reshape((-1, MAX_PLAYLIST_LENGTH, 1), input_shape=(MAX_PLAYLIST_LENGTH,)),
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    ngrams=2, 
                    vocabulary=vocab_dict['album_name_pl'],
                    name="album_name_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS, #  + 1, # 571625 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="album_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="album_name_pl_emb_layer_2d"),
            ], name="album_name_pl_emb_model"
        )
        
        # Feature: artist_genres_pl
        self.artist_genres_pl_embedding = tf.keras.Sequential(
            [
                # tf.keras.layers.Reshape((-1, MAX_PLAYLIST_LENGTH, 1), input_shape=(MAX_PLAYLIST_LENGTH,)),
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    ngrams=2, 
                    vocabulary=vocab_dict['artist_genres_pl'],
                    name="artist_genres_pl_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS, #  + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_genres_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_genres_pl_2d"),
            ], name="artist_genres_pl_emb_model"
        )

        # # Feature: tracks_playlist_titles_pl
        # self.tracks_playlist_titles_pl_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.TextVectorization(
        #             max_tokens=MAX_TOKENS, 
        #             ngrams=2, 
        #             # vocabulary=vocab_dict['tracks_playlist_titles_pl'],
        #             name="tracks_playlist_titles_pl_textvectorizor"
        #         ),
        #         tf.keras.layers.Embedding(
        #             input_dim=MAX_TOKENS + 1, 
        #             output_dim=EMBEDDING_DIM,
        #             name="tracks_playlist_titles_pl_emb_layer",
        #             mask_zero=False
        #         ),
        #         tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
        #         tf.keras.layers.GlobalAveragePooling2D(name="tracks_playlist_titles_pl_2d"),
        #     ], name="tracks_playlist_titles_pl_emb_model"
        # )
        
        # Feature: duration_ms_songs_pl
        self.duration_ms_songs_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(20744575)), # 20744575.0
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="duration_ms_songs_pl_emb_layer",
                    mask_zero=False
                ),
            tf.keras.layers.GlobalAveragePooling1D(name="duration_ms_songs_pl_emb_layer_pl_1d"),
            ], name="duration_ms_songs_pl_emb_model"
        )
        
        # Feature: track_pop_pl
        self.track_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_pop_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_pop_pl_1d"),
            ], name="track_pop_pl_emb_model"
        )
        
        # Feature: artist_pop_pl
        self.artist_pop_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_pop_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_pop_1d"),
            ], name="artist_pop_pl_emb_model"
        )
        
        # Feature: artists_followers_pl
        self.artists_followers_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(94437255)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artists_followers_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artists_followers_pl_1d"),
            ], name="artists_followers_pl_emb_model"
        )
        
        # Feature: track_danceability_pl
        self.track_danceability_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_danceability_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_danceability_pl_1d"),
            ], name="track_danceability_pl_emb_model"
        )
        
        # Feature: track_energy_pl
        self.track_energy_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_energy_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_energy_pl_1d"),
            ], name="track_energy_pl_emb_model"
        )
        
        # Feature: track_key_pl
        self.track_key_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=12), # , mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=12 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_key_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_key_pl_1d"),
            ], name="track_key_pl_emb_model"
        )
        
        # Feature: track_loudness_pl
        self.track_loudness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(5)), # TODO - Normalize? [-60, 5)
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
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
                    output_dim=EMBEDDING_DIM,
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
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_speechiness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_speechiness_pl_1d"),
            ], name="track_speechiness_pl_emb_model"
        )
        
        # Feature: track_acousticness_pl
        self.track_acousticness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_acousticness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_acousticness_pl_1d"),
            ], name="track_acousticness_pl_emb_model"
        )
        
        # Feature: track_instrumentalness_pl
        self.track_instrumentalness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_instrumentalness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_instrumentalness_pl_1d"),
            ], name="track_instrumentalness_pl_emb_model"
        )
        
        # Feature: track_liveness_pl
        self.track_liveness_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_liveness_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_liveness_pl_1d"),
            ], name="track_liveness_pl_emb_model"
        )
        
        # Feature: track_valence_pl
        self.track_valence_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_valence_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_valence_pl_1d"),
            ], name="track_valence_pl_emb_model"
        )
        
        # Feature: track_tempo_pl
        self.track_tempo_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(250)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
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
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="time_signature_pl_emb_layer",
                    input_shape=()
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="time_signature_pl_1d"),
            ], name="time_signature_pl_emb_model"
        )
        
        
        # ========================================
        # dense and cross layers
        # ========================================

        # Cross Layers
        if USE_CROSS_LAYER:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=PROJECTION_DIM,
                kernel_initializer="glorot_uniform", 
                name="pl_cross_layer"
            )
        else:
            self._cross_layer = None
            
        # Dense Layers
        self.dense_layers = tf.keras.Sequential(name="pl_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                )
            )
            if DROPOUT:
                self.dense_layers.add(tf.keras.layers.Dropout(DROPOUT_RATE))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
                )
            )
            
        ### ADDING L2 NORM AT THE END
        # self.dense_layers.add(tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, 1, epsilon=1e-12, name="normalize_dense")))
        # self.dense_layers.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x,0, epsilon=1e-12, name="normalize_dense")))
        self.dense_layers.add(tf.keras.layers.LayerNormalization(name="normalize_dense"))
        
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
                # self.pl_name_src_embedding(data['pl_name_src']),
                self.pl_name_src_text_embedding(data['pl_name_src']),
                self.pl_collaborative_src_embedding(data['pl_collaborative_src']),
                # self.num_pl_followers_src_embedding(data["num_pl_followers_src"]),
                self.pl_duration_ms_new_embedding(data["pl_duration_ms_new"]),
                self.num_pl_songs_new_embedding(data["num_pl_songs_new"]), # num_pl_songs_new | n_songs_pl_new
                self.num_pl_artists_new_embedding(data["num_pl_artists_new"]),
                self.num_pl_albums_new_embedding(data["num_pl_albums_new"]),
                # self.avg_track_pop_pl_new_embedding(data["avg_track_pop_pl_new"]),
                # self.avg_artist_pop_pl_new_embedding(data["avg_artist_pop_pl_new"]),
                # self.avg_art_followers_pl_new_embedding(data["avg_art_followers_pl_new"]),
                
                # sequence features
                self.track_uri_pl_embedding(data['track_uri_pl']),
                self.track_name_pl_embedding(tf.reshape(data['track_name_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.artist_uri_pl_embedding(data["artist_uri_pl"]),
                self.artist_name_pl_embedding(tf.reshape(data["artist_name_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.album_uri_pl_embedding(data["album_uri_pl"]),
                self.album_name_pl_embedding(tf.reshape(data["album_name_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.artist_genres_pl_embedding(tf.reshape(data["artist_genres_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                # self.tracks_playlist_titles_pl_embedding(tf.reshape(data["tracks_playlist_titles_pl"], [-1, MAX_PLAYLIST_LENGTH, 1])),
                
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
                self.time_signature_pl_embedding(data["time_signature_pl"]),
                
            ], axis=1)
        
        # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)

class Candidate_Track_Model(tf.keras.Model):
    def __init__(self, layer_sizes, vocab_dict):  
        super().__init__()
        
        # ========================================
        # Candidate features
        # ========================================
        
        # Feature: track_uri_can
        self.track_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2249561), # TODO
                tf.keras.layers.Embedding(
                    input_dim=2249561+1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_can_emb_layer",
                ),
            ], name="track_uri_can_emb_model"
        )
        
        # Feature: track_name_can
        self.track_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    vocabulary=vocab_dict['track_name_can'],
                    ngrams=2, 
                    name="track_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS+1,
                    output_dim=EMBEDDING_DIM,
                    name="track_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_name_can_1d"),
            ], name="track_name_can_emb_model"
        )
        
        # Feature: artist_uri_can
        self.artist_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=295860),
                tf.keras.layers.Embedding(
                    input_dim=295860+1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_uri_can_emb_layer",
                ),
            ], name="artist_uri_can_emb_model"
        )
        
        # Feature: artist_name_can
        self.artist_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    vocabulary=vocab_dict['artist_name_can'],
                    ngrams=2, 
                    name="artist_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS+1,
                    output_dim=EMBEDDING_DIM,
                    name="artist_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_name_can_1d"),
            ], name="artist_name_can_emb_model"
        )
        
        # Feature: album_uri_can
        self.album_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=734684),
                tf.keras.layers.Embedding(
                    input_dim=734684+1, 
                    output_dim=EMBEDDING_DIM,
                    name="album_uri_can_emb_layer",
                ),
            ], name="album_uri_can_emb_model"
        )

        # Feature: album_name_can
        self.album_name_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    vocabulary=vocab_dict['album_name_can'],
                    ngrams=2, 
                    name="album_name_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS+1,
                    output_dim=EMBEDDING_DIM,
                    name="album_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_name_can_1d"),
            ], name="album_name_can_emb_model"
        )
        
        # Feature: duration_ms_can
        self.duration_ms_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(20744575)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="duration_ms_can_emb_layer",
                ),
            ], name="duration_ms_can_emb_model"
        )
        
        # Feature: track_pop_can
        self.track_pop_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_pop_can_emb_layer",
                ),
            ], name="track_pop_can_emb_model"
        )
        
        # Feature: artist_pop_can
        self.artist_pop_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_pop_can_emb_layer",
                ),
            ], name="artist_pop_can_emb_model"
        )
        
        # Feature: artist_genres_can
        self.artist_genres_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(
                    # max_tokens=MAX_TOKENS, 
                    vocabulary=vocab_dict['artist_genres_can'],
                    ngrams=2, 
                    name="artist_genres_can_textvectorizor"
                ),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_genres_can_emb_layer",
                    mask_zero=True
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_can_1d"),
            ], name="artist_genres_can_emb_model"
        )
        
        # Feature: artist_followers_can
        self.artists_followers_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(94437255)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artists_followers_can_emb_layer",
                ),
            ], name="artists_followers_can_emb_model"
        )
        
        # Feature: track_pl_titles_can
        # self.track_pl_titles_can_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.TextVectorization(
        #             max_tokens=MAX_TOKENS, 
        #             ngrams=2, 
        #             # vocabulary=vocab_dict['track_pl_titles_can'],
        #             name="track_pl_titles_can_textvectorizor"
        #         ),
        #         tf.keras.layers.Embedding(
        #             input_dim=MAX_TOKENS + 1, 
        #             output_dim=EMBEDDING_DIM,
        #             name="track_pl_titles_can_emb_layer",
        #             mask_zero=False
        #         ),
        #         tf.keras.layers.GlobalAveragePooling1D(name="track_pl_titles_can_1d"),
        #     ], name="track_pl_titles_can_emb_model"
        # )
        
        # track_danceability_can
        self.track_danceability_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_danceability_can_emb_layer",
                ),
            ], name="track_danceability_can_emb_model"
        )
        
        # track_energy_can
        self.track_energy_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
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
                    output_dim=EMBEDDING_DIM,
                    name="track_key_can_emb_layer",
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="track_key_can_1d"),
            ], name="track_key_can_emb_model"
        )
        
        # track_loudness_can
        self.track_loudness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(5)), # TODO - Normalize? [-60, 5)
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
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
                    output_dim=EMBEDDING_DIM,
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
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_speechiness_can_emb_layer",
                ),
            ], name="track_speechiness_can_emb_model"
        )
        
        # track_acousticness_can
        self.track_acousticness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_acousticness_can_emb_layer",
                ),
            ], name="track_acousticness_can_emb_model"
        )
        
        # track_instrumentalness_can
        self.track_instrumentalness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_instrumentalness_can_emb_layer",
                ),
            ], name="track_instrumentalness_can_emb_model"
        )
        
        # track_liveness_can
        self.track_liveness_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_liveness_can_emb_layer",
                ),
            ], name="track_liveness_can_emb_model"
        )
        
        # track_valence_can
        self.track_valence_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(1)), # TODO - Normalize?
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_valence_can_emb_layer",
                ),
            ], name="track_valence_can_emb_model"
        )
        
        # track_tempo_can
        self.track_tempo_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(250)), # TODO - was 100
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
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
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="time_signature_can_emb_layer",
                    input_shape=()
                ),
                # tf.keras.layers.GlobalAveragePooling1D(name="time_signature_can_1d"),
            ], name="time_signature_can_emb_model"
        )
        
        # ========================================
        # Dense & Cross Layers
        # ========================================
        
        # Cross Layers
        if USE_CROSS_LAYER:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=PROJECTION_DIM,
                kernel_initializer="glorot_uniform", 
                name="can_cross_layer"
            )
        else:
            self._cross_layer = None
        
        # Dense Layer
        self.dense_layers = tf.keras.Sequential(name="candidate_dense_layers")
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED),
                )
            )
            if DROPOUT:
                self.dense_layers.add(tf.keras.layers.Dropout(DROPOUT_RATE))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED)
                )
            )
        ### ADDING L2 NORM AT THE END
        self.dense_layers.add(tf.keras.layers.LayerNormalization(name="normalize_dense"))
        
            
    # ========================================
    # Call Function
    # ========================================
            
    def call(self, data):

        all_embs = tf.concat(
            [
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

                # self.track_pl_titles_can_embedding(data['track_pl_titles_can']),
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
                self.time_signature_can_embedding(data['time_signature_can']),
            ], axis=1
        )
        
        # return self.dense_layers(all_embs)
                # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)


class TheTwoTowers(tfrs.models.Model):

    def __init__(self, layer_sizes, vocab_dict, parsed_candidate_dataset, compute_batch_metrics=False):   # vocab_dict
        super().__init__()
        
        self.query_tower = Playlist_Model(layer_sizes, vocab_dict)

        self.candidate_tower = Candidate_Track_Model(layer_sizes, vocab_dict)
        
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidate_dataset
                .batch(128)
                # .cache()
                .map(lambda x: (x['track_uri_can'], self.candidate_tower(x))), 
                ks=(10, 50, 100)),
            batch_metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(10, name='batch_categorical_accuracy_at_10'), 
                tf.keras.metrics.TopKCategoricalAccuracy(50, name='batch_categorical_accuracy_at_50')
            ],
            remove_accidental_hits=False,
            name="two_tower_retreival_task"
        )

        # self.task = tfrs.tasks.Retrieval(
        #     metrics=tfrs.metrics.FactorizedTopK(
        #         candidates=parsed_candidate_dataset.batch(128).map(
        #             self.candidate_tower,
        #             num_parallel_calls=tf.data.AUTOTUNE
        #         ).prefetch(tf.data.AUTOTUNE)
        #     )
        # )
                     
    def compute_loss(self, data, training=False):
        
        query_embeddings = self.query_tower(data)
        candidate_embeddings = self.candidate_tower(data)
        
        return self.task(
            query_embeddings, 
            candidate_embeddings, 
            compute_metrics=not training,
            candidate_ids=data['track_uri_can'],
            compute_batch_metrics=True
        ) # turn off metrics to save time on training
