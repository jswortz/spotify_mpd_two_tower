
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        # 'google-cloud-aiplatform==1.18.1',
        'google-cloud-storage',
        'tensorflow==2.8.3',
    ],
)
def adapt_ragged_text_layer_vocab(
    project: str,
    location: str,
    version: str,
    data_dir_bucket_name: str,
    data_dir_path_prefix: str,
    vocab_path_prefix: str,
    max_playlist_length: int,
    max_tokens: int,
    ngrams: int,
    feature_name: str,
    # feat_type: str,
) -> NamedTuple('Outputs', [
    ('vocab_gcs_uri', str),
    # ('feature_name', str),
]):

    """
    custom pipeline component to adapt the `pl_name_src` layer
    writes vocab to pickled dict in GCS
    dict combined with other layer vocabs and used in Two Tower training
    """
    
    # import packages
    import os
    import logging
    import pickle as pkl
    import time
    
    from google.cloud import storage
    
    import tensorflow as tf
    
    # setup clients
    storage_client = storage.Client()
    
    logging.info(f"feature_name: {feature_name}")
    # logging.info(f"feat_type: {feat_type}")
    
    MAX_PLAYLIST_LENGTH = max_playlist_length
    logging.info(f"MAX_PLAYLIST_LENGTH: {MAX_PLAYLIST_LENGTH}")
    
    # ===================================================
    # tfrecord parser
    # ===================================================
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
    
    # parsing function
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

    # list blobs (tfrecords)
    train_files = []
    for blob in storage_client.list_blobs(f'{data_dir_bucket_name}', prefix=f'{data_dir_path_prefix}'):
        if '.tfrecords' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
        
#     for blob in storage_client.list_blobs(f'{data_dir_bucket_name}', prefix=f'{data_dir_path_prefix}', delimiter="/"):
#         train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
        
#     # skip folder path prefix
#     train_files = train_files[1:]
    logging.info(f"TFRecord file count: {len(train_files)}")
    
    # ===================================================
    # create TF dataset
    # ===================================================
    logging.info(f"Creating TFRecordDataset...")
    train_dataset = tf.data.TFRecordDataset(train_files)
    train_parsed = train_dataset.map(parse_tfrecord)
    
    # ===================================================
    # adapt layer for feature
    # ===================================================
    
    # if feat_type == 'ragged':
    #     start = time.time()
    #     text_layer = tf.keras.layers.TextVectorization()
    #     text_layer.adapt(train_parsed.map(lambda x: tf.reshape(x[f'{feature_name}'], [-1, MAX_PLAYLIST_LENGTH, 1])))
    #     end = time.time()
    # else:
    #     start = time.time()
    #     text_layer = tf.keras.layers.TextVectorization()
    #     text_layer.adapt(train_parsed.map(lambda x: x[f'{feature_name}']))
    #     end = time.time()
    
    start = time.time()
    
    text_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        ngrams=ngrams
    )
    
    text_layer.adapt(train_parsed.map(lambda x: tf.reshape(x[f'{feature_name}'], [-1, MAX_PLAYLIST_LENGTH, 1])))
    end = time.time()

    logging.info(f'Layer adapt elapsed time: {round((end - start), 2)} seconds')
    
    # ===================================================
    # test layer
    # ===================================================
#     logging.info(f"Inspect {feature_name} layer's tokens and learned vocab...")
    
#     for row in train_parsed.batch(1).map(lambda x: x[f'{feature_name}']).take(1):
#       logging.info(f"{feature_name} tokens: {text_layer(row)}")
        
#     logging.info(f"{feature_name} vocab: {text_layer.get_vocabulary()[0:5]}")
    
    # ===================================================
    # write vocab to pickled dict --> gcs
    # ===================================================
    logging.info(f"Writting pickled dict to GCS...")
                 
    VOCAB_LOCAL_FILE = f'{feature_name}_vocab_dict.pkl'
    VOCAB_GCS_OBJ = f'{vocab_path_prefix}/{VOCAB_LOCAL_FILE}' # destination folder prefix and blob name
    VOCAB_DICT = {f'{feature_name}' : text_layer.get_vocabulary(),}
    
    logging.info(f"VOCAB_LOCAL_FILE: {VOCAB_LOCAL_FILE}")
    logging.info(f"VOCAB_GCS_OBJ: {VOCAB_GCS_OBJ}")

    # pickle
    filehandler = open(f'{VOCAB_LOCAL_FILE}', 'wb')
    pkl.dump(VOCAB_DICT, filehandler)
    filehandler.close()
    
    # upload to GCS
    bucket_client = storage_client.bucket(data_dir_bucket_name)
    blob = bucket_client.blob(VOCAB_GCS_OBJ)
    blob.upload_from_filename(VOCAB_LOCAL_FILE)
    
    vocab_uri = f'gs://{data_dir_bucket_name}/{VOCAB_GCS_OBJ}'
    
    logging.info(f"File {VOCAB_LOCAL_FILE} uploaded to {vocab_uri}")
    
    return(
        vocab_uri,
        # feature_name,
    )
