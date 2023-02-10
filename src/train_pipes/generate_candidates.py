
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.20.0',
        'tensorflow==2.10.1',
        'tensorflow-recommenders==0.7.2',
        # 'google-cloud-storage',
    ],
)
def generate_candidates(
    project: str,
    location: str,
    version: str, 
    # emb_index_gcs_uri: str,
    candidate_tower_dir_uri: str,
    candidate_file_dir_bucket: str,
    candidate_file_dir_prefix: str,
    experiment_run_dir: str,
) -> NamedTuple('Outputs', [
    ('emb_index_gcs_uri', str),
    # ('emb_index_artifact', Artifact),
]):
    import logging
    import json
    # import numpy as np
    import pickle as pkl
    from pprint import pprint
    import time

    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    import tensorflow as tf
    import tensorflow_recommenders as tfrs
    import tensorflow_io as tfio

    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

    import google.cloud.aiplatform as vertex_ai
    
    vertex_ai.init(
        project=project,
        location=location,
    )
    storage_client = storage.Client(project=project)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    # ====================================================
    # Load trained candidate tower
    # ====================================================
    
    logging.info(f"candidate_tower_dir_uri: {candidate_tower_dir_uri}")
    
    loaded_candidate_model = tf.saved_model.load(candidate_tower_uri)
    logging.info(f"loaded_candidate_model.signatures: {loaded_candidate_model.signatures}")
    
    candidate_predictor = loaded_candidate_model.signatures["serving_default"]
    logging.info(f"structured_outputs: {candidate_predictor.structured_outputs}")
    
    # ====================================================
    # Features and Helper Functions
    # ====================================================
    
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

    def full_parse(data):
        # used for interleave - takes tensors and returns a tf.dataset
        data = tf.data.TFRecordDataset(data)
        return data
    
    # ====================================================
    # Create Candidate Dataset
    # ====================================================

    candidate_files = []
    for blob in storage_client.list_blobs(f"{candidate_file_dir_bucket}", prefix=f'{candidate_file_dir_prefix}/'):
        if '.tfrecords' in blob.name:
            candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)

    parsed_candidate_dataset = candidate_dataset.interleave(
        # lambda x: tf.data.TFRecordDataset(x),
        full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).map(parse_candidate_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE).with_options(options)

    parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem
    
    # ====================================================
    # Generate embedding vectors for each candidate
    # ====================================================
    logging.info("Starting candidate dataset mapping...")
    
    start_time = time.time()

    embs_iter = parsed_candidate_dataset.batch(1).map(
        lambda data: candidate_predictor(
            track_uri_can = data["track_uri_can"],
            track_name_can = data['track_name_can'],
            artist_uri_can = data['artist_uri_can'],
            artist_name_can = data['artist_name_can'],
            album_uri_can = data['album_uri_can'],
            album_name_can = data['album_name_can'],
            duration_ms_can = data['duration_ms_can'],
            track_pop_can = data['track_pop_can'],
            artist_pop_can = data['artist_pop_can'],
            artist_genres_can = data['artist_genres_can'],
            artist_followers_can = data['artist_followers_can'],
            track_danceability_can = data['track_danceability_can'],
            track_energy_can = data['track_energy_can'],
            track_key_can = data['track_key_can'],
            track_loudness_can = data['track_loudness_can'],
            track_mode_can = data['track_mode_can'],
            track_speechiness_can = data['track_speechiness_can'],
            track_acousticness_can = data['track_acousticness_can'],
            track_instrumentalness_can = data['track_instrumentalness_can'],
            track_liveness_can = data['track_liveness_can'],
            track_valence_can = data['track_valence_can'],
            track_tempo_can = data['track_tempo_can'],
            time_signature_can = data['time_signature_can']
        )
    )

    embs = []
    for emb in embs_iter:
        embs.append(emb)

    end_time = time.time()
    elapsed_time = int((end_time - start_time) / 60)
    logging.info(f"elapsed_time: {elapsed_time}")

    logging.info(f"Length of embs: {len(embs)}")
    logging.info(f"embeddings[0]: {embs[0]}")
    
    # ====================================================
    # prep Track IDs and Vectors for JSON
    # ====================================================
    
    logging.info("Cleaning embeddings and track IDs...")
    start_time = time.time()
    cleaned_embs = [x['output_1'].numpy()[0] for x in embs] #clean up the output
    end_time = time.time()
    
    elapsed_time = int((end_time - start_time) / 60)
    logging.info(f"elapsed_time: {elapsed_time}")
    logging.info(f"Length of cleaned_embs: {len(cleaned_embs)}")
    
    # clean track IDs
    track_uris = [x['track_uri_can'].numpy() for x in parsed_candidate_dataset]
    logging.info(f"Length of track_uris: {len(track_uris)}")
    track_uris_decoded = [z.decode("utf-8") for z in track_uris]
    logging.info(f"Length of track_uris_decoded: {len(track_uris_decoded)}")
    
    # ZIP together
    logging.info("Zipping IDs and vectors ...")
    track_uris_valid = []
    emb_valid = []

    for i, pair in enumerate(zip(track_uris_decoded, cleaned_embs)):
        if i in bad_record_filter:
            pass
        else:
            t_uri, embed = pair
            track_uris_valid.append(t_uri)
            emb_valid.append(embed)
            
    # writting JSON file to GCS
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

    embeddings_index_filename = f'candidate_embs_{version}_{TIMESTAMP}.json'

    with open(f'{embeddings_index_filename}', 'w') as f:
        for prod, emb in zip(track_uris_valid, emb_valid):
            f.write('{"id":"' + str(prod) + '",')
            f.write('"embedding":[' + ",".join(str(x) for x in list(emb)) + "]}")
            f.write("\n")
            
    # write to GCS
    INDEX_GCS_URI = f'{experiment_run_dir}/candidate-embeddings-{version}'
    logging.info(f"INDEX_GCS_URI: {INDEX_GCS_URI}")

    DESTINATION_BLOB_NAME = embeddings_index_filename
    SOURCE_FILE_NAME = embeddings_index_filename

    logging.info(f"DESTINATION_BLOB_NAME: {DESTINATION_BLOB_NAME}")
    logging.info(f"SOURCE_FILE_NAME: {SOURCE_FILE_NAME}")
    
    blob = Blob.from_string(os.path.join(INDEX_GCS_URI, DESTINATION_BLOB_NAME))
    blob.bucket._client = storage_client
    blob.upload_from_filename(SOURCE_FILE_NAME)
    
    return (
        f'{INDEX_GCS_URI}',
        # f'{INDEX_GCS_URI}',
    )
