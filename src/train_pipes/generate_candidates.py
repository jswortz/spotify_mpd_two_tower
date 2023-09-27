
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (
    Artifact, Dataset, Input, InputPath, 
    Model, Output, OutputPath, component, Metrics
)
@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.26.1',
        'tensorflow==2.11.0',
        'tensorflow-recommenders==0.7.2',
        'numpy',
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
    train_output_gcs_bucket: str,
    experiment_name: str,
    experiment_run: str,
    experiment_run_dir: str,
) -> NamedTuple('Outputs', [
    ('emb_index_gcs_uri', str),
    # ('emb_index_artifact', Artifact),
]):
    import logging
    import json
    import pickle as pkl
    from pprint import pprint
    import time
    import numpy as np

    import os

    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    import tensorflow as tf
    import tensorflow_recommenders as tfrs

    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

    import google.cloud.aiplatform as vertex_ai
    
    # set clients
    vertex_ai.init(
        project=project,
        location=location,
    )
    storage_client = storage.Client(project=project)

    # tf.Data confg
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    # ====================================================
    # Load trained candidate tower
    # ====================================================
    logging.info(f"candidate_tower_dir_uri: {candidate_tower_dir_uri}")
    
    loaded_candidate_model = tf.saved_model.load(candidate_tower_dir_uri)
    logging.info(f"loaded_candidate_model.signatures: {loaded_candidate_model.signatures}")
    
    candidate_predictor = loaded_candidate_model.signatures["serving_default"]
    logging.info(f"structured_outputs: {candidate_predictor.structured_outputs}")
    
    # ===================================================
    # set feature vars
    # ===================================================
    FEATURES_PREFIX = f'{experiment_name}/{experiment_run}/features'
    logging.info(f"FEATURES_PREFIX: {FEATURES_PREFIX}")
    
    def download_blob(bucket_name, source_gcs_obj, local_filename):
        """Uploads a file to the bucket."""
        # storage_client = storage.Client(project=project_number)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_gcs_obj)
        blob.download_to_filename(local_filename)
        
        filehandler = open(f'{local_filename}', 'rb')
        loaded_dict = pkl.load(filehandler)
        filehandler.close()
        
        logging.info(f"File {local_filename} downloaded from gs://{bucket_name}/{source_gcs_obj}")
        
        return loaded_dict
    
    # ===================================================
    # load pickled Candidate features
    # ===================================================
    
    # candidate features
    CAND_FEAT_FILENAME = 'candidate_feats_dict.pkl'
    CAND_FEAT_GCS_OBJ = f'{FEATURES_PREFIX}/{CAND_FEAT_FILENAME}'
    LOADED_CANDIDATE_DICT = f'loaded_{CAND_FEAT_FILENAME}'
    
    loaded_candidate_features_dict = download_blob(
        train_output_gcs_bucket,
        CAND_FEAT_GCS_OBJ,
        LOADED_CANDIDATE_DICT
    )
    
    # ====================================================
    # Features and Helper Functions
    # ====================================================
    
    def parse_candidate_tfrecord_fn(example):
        """
        Reads candidate serialized examples from gcs and converts to tfrecord
        """
        # example = tf.io.parse_single_example(
        example = tf.io.parse_example(
            example, 
            features=loaded_candidate_features_dict
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
    
    embs_iter = parsed_candidate_dataset.batch(10000).map(
        lambda data: (
            data["track_uri_can"],
            loaded_candidate_model(data)
        )
    )
    
    embs = []
    for emb in embs_iter:
        embs.append(emb)

    end_time = time.time()
    elapsed_time = int((end_time - start_time) / 60)
    logging.info(f"elapsed_time   : {elapsed_time}")
    logging.info(f"Length of embs : {len(embs)}")
    logging.info(f"embeddings[0]  : {embs[0]}")
    
    # ====================================================
    # prep Track IDs and Vectors for JSON
    # ====================================================
    logging.info("Cleaning embeddings and track IDs...")
    start_time = time.time()
    
    # cleaned_embs = [x['output_1'].numpy()[0] for x in embs] #clean up the output
    
    cleaned_embs = []
    track_uris = []
    
    for ids , embedding in embs:
        cleaned_embs.extend(embedding.numpy())
        track_uris.extend(ids.numpy())
    
    end_time = time.time()
    elapsed_time = int((end_time - start_time) / 60)
    logging.info(f"elapsed_time           : {elapsed_time}")
    logging.info(f"Length of cleaned_embs : {len(cleaned_embs)}")
    logging.info(f"Length of track_uris: {len(track_uris)}")
    
    track_uris_decoded = [z.decode("utf-8") for z in track_uris]
    logging.info(f"Length of track_uris decoded: {len(track_uris_decoded)}")
    logging.info(f"track_uris_decoded[0]       : {track_uris_decoded[0]}")
    
    # check for bad records
    bad_records = []

    for i, emb in enumerate(cleaned_embs):
        bool_emb = np.isnan(emb)
        for val in bool_emb:
            if val:
                bad_records.append(i)

    bad_record_filter = np.unique(bad_records)

    logging.info(f"bad_records: {len(bad_records)}")
    logging.info(f"bad_record_filter: {len(bad_record_filter)}")
    
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
            
    logging.info(f"track_uris_valid[0]: {track_uris_valid[0]}")
    logging.info(f"bad_records: {len(bad_records)}")
            
    # ====================================================
    # writting JSON file to GCS
    # ====================================================
    TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
    embeddings_index_filename = f'candidate_embs.json'

    with open(f'{embeddings_index_filename}', 'w') as f:
        for prod, emb in zip(track_uris_valid, emb_valid):
            f.write('{"id":"' + str(prod) + '",')
            f.write('"embedding":[' + ",".join(str(x) for x in list(emb)) + "]}")
            f.write("\n")
            
    # write to GCS
    INDEX_GCS_URI = f'{experiment_run_dir}/candidate-embeddings-{TIMESTAMP}'
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
