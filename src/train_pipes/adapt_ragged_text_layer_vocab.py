
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        # 'google-cloud-aiplatform==1.18.1',
        'google-cloud-storage',
        'tensorflow==2.10.1',
    ],
)
def adapt_ragged_text_layer_vocab(
    project: str,
    location: str,
    version: str,
    data_dir_bucket_name: str,
    data_dir_path_prefix: str,
    train_output_gcs_bucket: str,
    experiment_name: str,
    experiment_run: str,
    max_playlist_length: int,
    max_tokens: int,
    ngrams: int,
    feature_name: str,
    generate_new_vocab: bool,
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
    
    storage_client = storage.Client(project=project)
    
    logging.info(f"feature_name: {feature_name}")
    # logging.info(f"feat_type: {feat_type}")
    
    # ===================================================
    # set feature vars
    # ===================================================
    MAX_PLAYLIST_LENGTH = max_playlist_length
    logging.info(f"MAX_PLAYLIST_LENGTH: {MAX_PLAYLIST_LENGTH}")
    
    FEATURES_PREFIX = f'{experiment_name}/{experiment_run}/features'
    logging.info(f"FEATURES_PREFIX: {FEATURES_PREFIX}")
    
    all_features_dict = {}
    
    # ===================================================
    # load pickled Candidate features
    # ===================================================
    
    # candidate features
    CAND_FEAT_FILENAME = 'candidate_feats_dict.pkl'
    CAND_FEAT_GCS_OBJ = f'{FEATURES_PREFIX}/{CAND_FEAT_FILENAME}'
    LOADED_CANDIDATE_DICT = f'loaded_{CAND_FEAT_FILENAME}'
    logging.info(f"CAND_FEAT_FILENAME: {CAND_FEAT_FILENAME}; CAND_FEAT_GCS_OBJ:{CAND_FEAT_GCS_OBJ}; LOADED_CANDIDATE_DICT: {LOADED_CANDIDATE_DICT}")
    
    # os.system(f'gsutil cp gs://{train_output_gcs_bucket}/{CAND_FEAT_GCS_OBJ} {LOADED_CANDIDATE_DICT}')
    bucket = storage_client.bucket(train_output_gcs_bucket)
    blob = bucket.blob(CAND_FEAT_GCS_OBJ)
    blob.download_to_filename(LOADED_CANDIDATE_DICT)
    
    filehandler = open(f'{LOADED_CANDIDATE_DICT}', 'rb')
    loaded_candidate_features_dict = pkl.load(filehandler)
    filehandler.close()
    logging.info(f"loaded_candidate_features_dict: {loaded_candidate_features_dict}")
    
    all_features_dict.update(loaded_candidate_features_dict)
    logging.info(f"all_features_dict: {all_features_dict}")

    # ===================================================
    # load pickled Query features
    # ===================================================

    # query features
    QUERY_FEAT_FILENAME = 'query_feats_dict.pkl'
    QUERY_FEAT_GCS_OBJ = f'{FEATURES_PREFIX}/{QUERY_FEAT_FILENAME}'
    LOADED_QUERY_DICT = f'loaded_{QUERY_FEAT_FILENAME}'
    logging.info(f"QUERY_FEAT_FILENAME: {QUERY_FEAT_FILENAME}; QUERY_FEAT_GCS_OBJ:{QUERY_FEAT_GCS_OBJ}; LOADED_QUERY_DICT: {LOADED_QUERY_DICT}")
    
    # os.system(f'gsutil cp gs://{train_output_gcs_bucket}/{QUERY_FEATURES_GCS_OBJ} {LOADED_QUERY_DICT}')
    bucket = storage_client.bucket(train_output_gcs_bucket)
    blob = bucket.blob(QUERY_FEAT_GCS_OBJ)
    blob.download_to_filename(LOADED_QUERY_DICT)
    
    filehandler = open(f'{LOADED_QUERY_DICT}', 'rb')
    loaded_query_features_dict = pkl.load(filehandler)
    filehandler.close()
    logging.info(f"loaded_query_features_dict: {loaded_query_features_dict}")
    
    all_features_dict.update(loaded_query_features_dict)
    logging.info(f"all_features_dict: {all_features_dict}")
    
    # ===================================================
    # tfrecord parser
    # ===================================================
    
    # parsing function
    def parse_tfrecord(example):
        """
        Reads a serialized example from GCS and converts to tfrecord
        """
        # example = tf.io.parse_single_example(
        example = tf.io.parse_example(
            example,
            # feats
            features=all_features_dict
        )
        return example
    
    
    if generate_new_vocab:
        logging.info(f"Generating new vocab file...")
    
        # list blobs (tfrecords)
        train_files = []
        for blob in storage_client.list_blobs(f'{data_dir_bucket_name}', prefix=f'{data_dir_path_prefix}'):
            if '.tfrecords' in blob.name:
                train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

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

        start = time.time()
        text_layer = tf.keras.layers.TextVectorization(
            max_tokens=max_tokens,
            ngrams=ngrams
        )
        text_layer.adapt(train_parsed.map(lambda x: tf.reshape(x[f'{feature_name}'], [-1, MAX_PLAYLIST_LENGTH, 1])))
        end = time.time()

        logging.info(f'Layer adapt elapsed time: {round((end - start), 2)} seconds')

        # ===================================================
        # write vocab to pickled dict --> gcs
        # ===================================================
        logging.info(f"Writting pickled dict to GCS...")

        VOCAB_LOCAL_FILE = f'{feature_name}_vocab_dict.pkl'
        VOCAB_GCS_OBJ = f'{experiment_name}/{experiment_run}/vocab-staging/{VOCAB_LOCAL_FILE}' # destination folder prefix and blob name
        VOCAB_DICT = {f'{feature_name}' : text_layer.get_vocabulary(),}

        logging.info(f"VOCAB_LOCAL_FILE: {VOCAB_LOCAL_FILE}")
        logging.info(f"VOCAB_GCS_OBJ: {VOCAB_GCS_OBJ}")

        # pickle
        filehandler = open(f'{VOCAB_LOCAL_FILE}', 'wb')
        pkl.dump(VOCAB_DICT, filehandler)
        filehandler.close()

        # upload to GCS
        bucket_client = storage_client.bucket(train_output_gcs_bucket)
        blob = bucket_client.blob(VOCAB_GCS_OBJ)
        blob.upload_from_filename(VOCAB_LOCAL_FILE)

        vocab_uri = f'gs://{train_output_gcs_bucket}/{VOCAB_GCS_OBJ}'

        logging.info(f"File {VOCAB_LOCAL_FILE} uploaded to {vocab_uri}")
        
    else:
        logging.info(f"Using existing vocab files...")
        vocab_uri = 'gs://two-tower-models/vocabs/vocab_dict.pkl'
        logging.info(f"Using vocab file: {vocab_uri}")
    
    return(
        vocab_uri,
        # feature_name,
    )
    