import json
import tensorflow as tf
import tensorflow_recommenders as tfrs

from google.cloud import storage


import numpy as np
import pickle as pkl
import os
from pprint import pprint

MAX_PLAYLIST_LENGTH = 5 # this is set upstream by the BigQuery max length
EMBEDDING_DIM = 128
PROJECTION_DIM = 100
SEED = 1234
USE_CROSS_LAYER=False
DROPOUT=False
DROPOUT_RATE=0.33
MAX_TOKENS=50000

client = storage.Client()

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    # bucket_name = bucket_name.strip("gs://")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


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
    'track_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    'artist_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    'album_name_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
    'track_uri_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH)),
    'duration_ms_songs_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'artist_pop_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'artists_followers_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'track_pop_pl': tf.io.FixedLenFeature(dtype=tf.float32, shape=(MAX_PLAYLIST_LENGTH,)),
    'artist_genres_pl': tf.io.FixedLenFeature(dtype=tf.string, shape=(MAX_PLAYLIST_LENGTH,)),
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


BUCKET = 'spotify-beam-v3'
CANDIDATE_PREFIX = 'v1/candidates/'

candidate_files = []
# for blob in client.list_blobs(f"{BUCKET}", prefix=f'{CANDIDATE_PREFIX}', delimiter="/"):
candidate_files.append('candidates-00000-of-00001.tfrecords')

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
).with_options(options)

parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem

client = storage.Client()
def get_buckets_20(MAX_VAL):
    """ This helper funciton
        creates discretization buckets of size 20
        MAX_VAL: the max value of the tensor to be discritized
        (Assuming 1 is min value)
    """
    list_buckets = list(np.linspace(1, MAX_VAL, num=20))
    return(list_buckets)
    
import pickle as pkl
os.system('gsutil cp gs://two-tower-models/vocabs/vocab_dict.pkl .')

filehandler = open('vocab_dict.pkl', 'rb')
vocab_dict = pkl.load(filehandler)
filehandler.close()

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
                # tf.keras.layers.GlobalAveragePooling1D(name="pl_name_pooling"),
            ], name="pl_name_emb_model"
        )
        
        # Feature: collaborative
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
        
        # ========================================
        # sequence playlist features
        # ========================================
        
        # Feature: artist_name_pl
        self.artist_name_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  ngrams=2, 
                                                  vocabulary=vocab_dict['artist_name_pl'],
                                                  name="artist_name_pl_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_name_pl_2d"),
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
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_uri_1d"),
            ], name="track_uri_pl_emb_model"
        )
        
        # Feature: track_name_pl
        self.track_name_pl_embedding = tf.keras.Sequential(
            [
            tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  ngrams=2, 
                                              vocabulary=vocab_dict['track_name_pl'],
                                              name="track_name_pl_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_name_pl_emb_layer",
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="track_name_pl_2d"),
            ], name="track_name_pl_emb_model"
        )
        
    
        # self.duration_ms_songs_pl_embedding = tf.keras.Sequential(
        #     [
        #         tf.keras.layers.Discretization(get_buckets_20(20744575)),
        #         tf.keras.layers.Embedding(
        #             input_dim=20 + 1, 
        #             output_dim=EMBEDDING_DIM,
        #             name="duration_ms_songs_pl_emb_layer",
        #             mask_zero=False
        #         ),
        #     tf.keras.layers.GlobalAveragePooling1D(name="duration_ms_songs_pl_emb_layer_pl_1d"),
        #     ], name="duration_ms_songs_pl_emb_model"
        # )
        
        # Feature: album_name_pl
        self.album_name_pl_embedding = tf.keras.Sequential(
            [
            tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  ngrams=2, 
                                              vocabulary=vocab_dict['album_name_pl'],
                                              name="album_name_pl_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=571625 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="album_name_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="album_name_pl_emb_layer_2d"),
            ], name="album_name_pl_emb_model"
        )
        
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
        
        self.artists_followers_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artists_followers_pl_emb_layer",
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artists_followers_pl_1d"),
            ], name="artists_followers_pl_emb_model"
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
        
        # Feature: artist_genres_pl
        self.artist_genres_pl_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  ngrams=2, 
                                                  vocabulary=vocab_dict['artist_genres_pl'],
                                                  name="artist_genres_pl_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_genres_pl_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.Reshape([-1, MAX_PLAYLIST_LENGTH, EMBEDDING_DIM]),
                tf.keras.layers.GlobalAveragePooling2D(name="artist_genres_pl_2d"),
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
        self.dense_layers.add(tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, 1, epsilon=1e-12, name="normalize_dense")))
        
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
                
                # sequence features
                self.artist_name_pl_embedding(tf.reshape(data['artist_name_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.track_uri_pl_embedding(data["track_uri_pl"]),
                self.track_name_pl_embedding(tf.reshape(data['track_name_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.album_name_pl_embedding(tf.reshape(data['album_name_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
                self.artist_pop_pl_embedding(data["artist_pop_pl"]),
                self.artists_followers_pl_embedding(data["artists_followers_pl"]),
                self.track_pop_pl_embedding(data["track_pop_pl"]),
                self.artist_genres_pl_embedding(tf.reshape(data['artist_genres_pl'], [-1, MAX_PLAYLIST_LENGTH, 1])),
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
                tf.keras.layers.Hashing(num_bins=287710),
                tf.keras.layers.Embedding(
                    input_dim=287710+1,
                    output_dim=EMBEDDING_DIM,
                    name="artist_name_can_emb_layer",
                ),
            ], name="artist_name_can_emb_model"
        )
        
#         # Feature: track_name_can
        self.track_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  vocabulary=vocab_dict['track_name_can'],
                                                  ngrams=2, name="track_name_can_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS+1,
                    output_dim=EMBEDDING_DIM,
                    name="track_name_can_emb_layer",
                    mask_zero=True
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="track_name_can_1d"),
            ], name="track_name_can_emb_model"
        )
        
        # Feature: album_name_can
        self.album_name_can_text_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  vocabulary=vocab_dict['album_name_can'],
                                                  ngrams=2, name="album_name_can_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS+1,
                    output_dim=EMBEDDING_DIM,
                    name="album_name_can_emb_layer",
                    mask_zero=False
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="album_name_can_1d"),
            ], name="album_name_can_emb_model"
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
        
        # Feature: track_uri_can
        self.track_uri_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Hashing(num_bins=2262292),
                tf.keras.layers.Embedding(
                    input_dim=2262292+1, 
                    output_dim=EMBEDDING_DIM,
                    name="track_uri_can_emb_layer",
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
                ),
            ], name="album_uri_can_emb_model"
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
        
        # Feature: artist_followers_can
        self.artists_followers_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Discretization(get_buckets_20(100)),
                tf.keras.layers.Embedding(
                    input_dim=20 + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artists_followers_can_emb_layer",
                ),
            ], name="artists_followers_can_emb_model"
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
        
        # Feature: artist_genres_can
        self.artist_genres_can_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.TextVectorization(max_tokens=MAX_TOKENS, 
                                                  vocabulary=vocab_dict['artist_genres_can'],
                                                  ngrams=2, name="artist_genres_can_textvectorizor"),
                tf.keras.layers.Embedding(
                    input_dim=MAX_TOKENS + 1, 
                    output_dim=EMBEDDING_DIM,
                    name="artist_genres_can_emb_layer",
                    mask_zero=True
                ),
                tf.keras.layers.GlobalAveragePooling1D(name="artist_genres_can_1d"),
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
        self.dense_layers.add(tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, 1, epsilon=1e-12, name="normalize_dense")))
            
    # ========================================
    # Call Function
    # ========================================
            
    def call(self, data):

        all_embs = tf.concat(
            [
                self.track_name_can_text_embedding(data['track_name_can']),  
                self.album_name_can_text_embedding(data['album_name_can']),  
                self.artist_uri_can_embedding(data['artist_uri_can']),  
                self.track_uri_can_embedding(data['track_uri_can']),  
                self.album_uri_can_embedding(data['album_uri_can']), 
                self.artist_name_can_text_embedding(data['artist_name_can']),
                self.duration_ms_can_embedding(data["duration_ms_can"]), 
                self.track_pop_can_embedding(data["track_pop_can"]),  
                self.artist_pop_can_embedding(data["artist_pop_can"]),  
                self.artists_followers_can_embedding(data["artist_followers_can"]),  
                self.artist_genres_can_embedding(data['artist_genres_can']),  
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
        
        self.__metrics = tfrs.metrics.FactorizedTopK(
                candidates=parsed_candidate_dataset.batch(2048).map(self.candidate_tower))
        self.__metrics.reset_states()
        self.task = tfrs.tasks.Retrieval(
                    metrics=self.__metrics,
                    num_hard_negatives=100, #number of candidates to consider sorted by max logits
                    # remove_accidental_hits=True, #remove the candidate from the negative samples if it accidentally is in the list
                    name="two_tower_retreival_task")
                
        
    def compute_loss(self, data, training=False):
        query_embeddings = self.query_tower(data)
        candidate_embeddings = self.candidate_tower(data)

        return self.task(
            query_embeddings, 
            candidate_embeddings, 
            compute_metrics=not training
        ) # turn off metrics to save time on training
    
###APPENDIX HELPER DIAGNOSTIC FUNCTIONS FOR NULLS:

# for i in range(3,8):
#     for _ in train_dataset.skip(i).take(1):#.map(lambda data:tf.reshape(data['track_name_pl'], [-1, 5, 1])).take(1):
#         res = model.candidate_tower.layers[1](_['track_name_can'])
#         v = tf.where(tf.math.is_nan(res), 1., 0.)
#         print(tf.reduce_max(v).numpy()) #if this returns a one we have nans
#     #     # ALBUM NAME CAN: tf.Tensor(1.0, shape=(), dtype=float32)
#     # TRACK NAME CAN - has nulls with ALBUM NAME CAN

    
# #     Track (candidate) Tower:
# # 0 artist_name_can_emb_model
# # 1 track_name_can_emb_model
# # 2 artist_uri_can_emb_model
# # 3 track_uri_can_emb_model
# # 4 album_uri_can_emb_model
# # 5 duration_ms_can_emb_model
# # 6 artist_pop_can_emb_model
# # 7 artists_followers_can_emb_model
# # 8 track_pop_can_emb_model
# # 9 artist_genres_can_emb_model
# # 10 candidate_dense_layers


# # # Playlist (query) Tower:
# # # 0 pl_name_emb_model
# # # 1 pl_collaborative_emb_model
# # # 2 pl_track_uri_emb_model
# # # 3 artist_name_pl_emb_model
# # # 4 track_uri_pl_emb_model
# # # 5 track_name_pl_emb_model
# # # 6 duration_ms_songs_pl_emb_model
# # # 7 artist_pop_pl_emb_model
# # # 8 artists_followers_pl_emb_model
# # # 9 track_pop_pl_emb_model
# # # 10 artist_genres_pl_emb_model
# # # 11 pl_dense_layer