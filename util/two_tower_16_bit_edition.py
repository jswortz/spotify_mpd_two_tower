import json
import tensorflow as tf
import tensorflow_recommenders as tfrs

from google.cloud import storage
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

import numpy as np
import pickle as pkl
from pprint import pprint

MAX_PLAYLIST_LENGTH = 5 # this is set upstream by the BigQuery max length
EMBEDDING_DIM = 128
PROJECTION_DIM = 100
SEED = 1234
USE_CROSS_LAYER=True
DROPOUT=True
DROPOUT_RATE=0.33

client = storage.Client()
candidate_features = {
    'track_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'duration_ms_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'track_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_genres_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_followers_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
}

feats = {
    'track_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_name_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'album_uri_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'duration_ms_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'track_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_pop_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'artist_genres_can': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'artist_followers_can': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'name': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'collaborative': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'n_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_artists_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'num_albums_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    'description_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    'track_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
    'artist_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
    'album_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
    'track_uri_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
    'duration_ms_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH)),
    'artist_pop_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH)),
    'artists_followers_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH)),
    'track_pop_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH)),
    'artist_genres_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
}

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO


def parse_tfrecord(example):
    """
    Reads a serialized example from GCS and converts to tfrecord
    """
    example = tf.io.parse_single_example(
        example, 
        feats
    )
    return example


def parse_candidate_tfrecord_fn(example):
    """
    Reads candidate serialized examples from gcs and converts to tfrecord
    """
    example = tf.io.parse_single_example(
        example, 
        features=candidate_features
    )
    return example

BUCKET_NAME = 'spotify-v1'
# FILE_PATH = 'vocabs/v2_string_vocabs'
# FILE_NAME = 'string_vocabs_v1_20220924-tokens22.pkl'
# DESTINATION_FILE = 'downloaded_vocabs.txt'

BUCKET = 'spotify-beam-v3'
CANDIDATE_PREFIX = 'v3/candidates/'

candidate_files = []
for blob in client.list_blobs(f"{BUCKET}", prefix=f'{CANDIDATE_PREFIX}', delimiter="/"):
    candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

#generate the candidate dataset

candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)
parsed_candidate_dataset = candidate_dataset.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
).map(
    parse_candidate_tfrecord_fn,
    num_parallel_calls=tf.data.AUTOTUNE,
).with_options(options).cache()


client = storage.Client()
    
class Playlist_Model(tf.keras.Model):
    def __init__(self, layer_sizes):
        super().__init__()

        # ========================================
        # non-sequence playlist features
        # ========================================
        
        # Feature: playlist name
        self.pl_name_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=1_000_000), #one MILLION playlists
                tf.keras.layers.Embedding(
                    input_dim=1_000_000 + 1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="pl_name_emb_layer",
                    input_shape=()
                ),
            ], name="pl_name_emb_model"
        )
        
        # Feature: collaborative
#         collaborative_vocab = np.array([b'false', b'true'])
        
        self.pl_collaborative_embedding = tf.keras.Sequential(
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
        
        # Feature: pid
        self.pl_track_uri_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2262292),

                tf.keras.layers.Embedding(
                    input_dim=2262292+1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="pl_track_uri_layer",
                    input_shape=()
                ),
            ], name="pl_track_uri_emb_model"
        )
        
        # Feature: n_songs_pl
        self.n_songs_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=58.6523,
            variance=2275.8927,
            axis=None,
            name="n_songs_pl_normalized"
        )
        
        # Feature: num_artists_pl
        self.n_artists_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=32.42,
            variance=763,
            axis=None,
            name="n_artists_pl_normalized"
        )

        # Feature: num_albums_pl
        self.n_albums_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=42.73,
            variance=1290,
            axis=None,
            name="n_albums_pl_normalized"
        )
        
        # ========================================
        # sequence playlist features
        # ========================================
        
        # Feature: artist_name_pl
        self.artist_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=287710, mask_value=''),

                tf.keras.layers.Embedding(
                    input_dim=287710 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_name_pl_emb_layer",
                    mask_zero=True,
                    input_shape=(None,)
                    
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_name_pl_1d"),
            ], name="artist_name_pl_emb_model"
        )
        
        # Feature: track_uri_pl
        # 2.2M unique
        self.track_uri_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2262292, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=2262292 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_pl_emb_layer",
                    mask_zero=True,
                    input_shape=(None,)
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_uri_1d"),
            ], name="track_uri_pl_emb_model"
        )
        
        # Feature: track_name_pl
        self.track_name_pl_embedding = tf.keras.Sequential(
            [
            tf.keras.layers.Hashing(num_bins=1483753, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=1483753 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_name_pl_emb_layer",
                    mask_zero=True,
                    input_shape=(None,)
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_name_pl_1d"),
            ], name="track_name_pl_emb_model"
        )
        
        #Feature: duration_ms_songs_pl
        self.duration_ms_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=234762.99476987208,
            variance=5411028799.28701,
            axis=None,
            name="duration_ms_pl_normalized"
        )
        
        # Feature: album_name_pl
        self.album_name_pl_embedding = tf.keras.Sequential(
            [
            tf.keras.layers.Hashing(num_bins=571625, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=571625+1 , 
                    output_dim=EMBEDDING_DIM,
                    name="album_name_pl_emb_layer",
                    mask_zero=True,
                    input_shape=(None,)
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_name_pl_emb_layer_1d"),
            ], name="album_name_pl_emb_model"
        )
        
        # Feature: artist_pop_pl
        self.artist_pop_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=16.08,
            variance=300.64,
            axis=None,
            name="artist_pop_pl_normalized"
        )
        
        # Feature: artists_followers_pl
        self.artist_followers_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=7045512.193,
            variance=178225831161684.75,
            axis=None,
            name="artist_followers_pl_normalized"
        )
        
        # Feature: track_pop_pl
        self.track_pop_pl_normalized = tf.keras.layers.experimental.preprocessing.Normalization(
            mean=38.937,
            variance=922,
            axis=None,
            name="track_pop_pl_normalized"
        )
        
        # Feature: artist_genres_pl
        self.artist_genres_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=734684, mask_value=''),
                tf.keras.layers.Embedding(
                    input_dim=734684 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_genres_pl_emb_layer",
                    mask_zero=True,
                    input_shape=(None,)
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_pl_1d"),
            ], name="artist_genres_pl_emb_model"
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
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=initializer,
                )
            )
            if DROPOUT:
                self.dense_layers.add(tf.keras.layers.Dropout(DROPOUT_RATE))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=initializer
                )
            )
        ### ADDING L2 NORM AT THE END
        self.dense_layers.add(
            tf.keras.layers.Lambda(
                lambda x: tf.nn.l2_normalize(
                    x, 1, epsilon=1e-12, name="normalize_dense"
                )
            )
        )
        
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
                self.pl_name_text_embedding(data['name']),
                self.pl_collaborative_embedding(data['collaborative']),
                self.pl_track_uri_embedding(data["track_uri_can"]),
                tf.reshape(self.n_songs_pl_normalized(data["n_songs_pl"]), (-1, 1)),
                tf.reshape(self.n_artists_pl_normalized(data['num_artists_pl']), (-1, 1)),
                tf.reshape(self.n_albums_pl_normalized(data["num_albums_pl"]), (-1, 1)),
                
                # sequence features
                self.artist_name_pl_embedding(data["artist_name_pl"]),
                self.track_uri_pl_embedding(data["track_uri_pl"]),
                self.track_name_pl_embedding(data["track_name_pl"]),
                self.duration_ms_pl_normalized(data["duration_ms_songs_pl"]),
                self.album_name_pl_embedding(data["album_name_pl"]),
                self.artist_pop_pl_normalized(data["artist_pop_pl"]),
                self.artist_followers_pl_normalized(data["artists_followers_pl"]),
                self.track_pop_pl_normalized(data["track_pop_pl"]),
                self.artist_genres_pl_embedding(data["artist_genres_pl"]),
            ], axis=1)
        
        # Build Cross Network
        if self._cross_layer is not None:
            cross_embs = self._cross_layer(all_embs)
            return self.dense_layers(cross_embs)
        else:
            return self.dense_layers(all_embs)

class Candidate_Track_Model(tf.keras.Model):
    def __init__(self, layer_sizes):
        super().__init__()
        
        # ========================================
        # Candidate features
        # ========================================
        
        # Feature: artist_name_can
        self.artist_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim=200_000+1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="artist_name_can_emb_layer",
                    input_shape=()
                ),
            ], name="artist_name_can_emb_model"
        )
        
        # Feature: track_name_can
        self.track_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim=200_000+1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="track_name_can_emb_layer",
                    input_shape=()
                ),
            ], name="track_name_can_emb_model"
        )
        
        # Feature: album_name_can
        self.album_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim=200_000+1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="album_name_can_emb_layer",
                    input_shape=()
                ),
            ], name="album_name_can_emb_model"
        )
        
        # Feature: artist_uri_can
        self.artist_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim=200_000+1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_uri_can_emb_layer",
                    input_shape=()
                ),
            ], name="artist_uri_can_emb_model"
        )
        
        # Feature: track_uri_can
        self.track_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2262292),
                tf.keras.layers.Embedding(
                    input_dim=2262292+1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_can_emb_layer",
                    input_shape=()
                ),
            ], name="track_uri_can_emb_model"
        )
        
        # Feature: album_uri_can
        self.album_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=734684),
                tf.keras.layers.Embedding(
                    input_dim=734684+1, 
                    output_dim=EMBEDDING_DIM,
                    name="album_uri_can_emb_layer",
                    input_shape=()
                ),
            ], name="album_uri_can_emb_model"
        )
        
        # Feature: duration_ms_can
        self.duration_ms_can_normalized = tf.keras.layers.Normalization(
            mean=234740.855138,
            variance=5456526884.81,
            axis=None,
            name="duration_ms_can_normalized"
        )
        
        # Feature: track_pop_can
        self.track_pop_can_normalized = tf.keras.layers.Normalization(
            mean=38.86502,
            variance=921,
            axis=None,
            name="track_pop_can_normalized"
        )
        
        # Feature: artist_pop_can
        self.artist_pop_can_normalized = tf.keras.layers.Normalization(
            mean=16.08,
            variance=300.64,
            axis=None,
            name="artist_pop_can_normalized"
        )
        
        # Feature: artist_followers_can
        self.artist_followers_can_normalized = tf.keras.layers.Normalization(
            mean=7002679.6240426032,
            variance=176927172310928.59,
            axis=None,
            name="artist_followers_can_normalized"
        )
        
        # Feature: artist_genres_can
        self.artist_genres_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=200_000),
                tf.keras.layers.Embedding(
                    input_dim=200_000+1,
                    output_dim=EMBEDDING_DIM,
                    mask_zero=False,
                    name="artist_genres_can_emb_layer",
                    input_shape=()
                ),
            ], name="artist_genres_can_emb_model"
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
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        
        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    activation="relu", 
                    kernel_initializer=initializer,
                )
            )
            if DROPOUT:
                self.dense_layers.add(tf.keras.layers.Dropout(DROPOUT_RATE))
                
        # No activation for the last layer
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(
                tf.keras.layers.Dense(
                    layer_size, 
                    kernel_initializer=initializer
                )
            )
            
    # ========================================
    # Call Function
    # ========================================
            
    def call(self, data):
        
        all_embs = tf.concat(
            [
                self.artist_name_can_text_embedding(data['artist_name_can']),  
                self.track_name_can_text_embedding(data['track_name_can']),  
                self.album_name_can_text_embedding(data['album_name_can']),  
                self.artist_uri_can_embedding(data['artist_uri_can']),  
                self.track_uri_can_embedding(data['track_uri_can']),  
                self.album_uri_can_embedding(data['album_uri_can']),  
                tf.reshape(self.duration_ms_can_normalized(data["duration_ms_can"]), (-1, 1)), 
                tf.reshape(self.track_pop_can_normalized(data["track_pop_can"]), (-1, 1)),  
                tf.reshape(self.artist_pop_can_normalized(data["artist_pop_can"]), (-1, 1)),  
                tf.reshape(self.artist_followers_can_normalized(data["artist_followers_can"]), (-1, 1)),  
                self.artist_genres_can_text_embedding(data['artist_genres_can']),  
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

    def __init__(self, layer_sizes ):
        super().__init__()
        self.query_tower = Playlist_Model(layer_sizes)

        self.candidate_tower = Candidate_Track_Model(layer_sizes)

        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidate_dataset.batch(512).map(self.candidate_tower, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE,)))
        
    def compute_loss(self, data, training=False):
        query_embeddings = self.query_tower(data)
        candidate_embeddings = self.candidate_tower(data)

        return self.task(
            query_embeddings, 
            candidate_embeddings, 
            compute_metrics=not training
        ) # turn off metrics to save time on training