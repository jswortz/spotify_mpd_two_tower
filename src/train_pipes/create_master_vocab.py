
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        # 'google-cloud-aiplatform==1.18.1',
        'google-cloud-storage',
        'numpy',
        # 'tensorflow==2.8.3',
    ],
)
def create_master_vocab(
    project: str,
    location: str,
    version: str,
    train_output_gcs_bucket: str,
    experiment_name: str,
    experiment_run: str,
    vocab_uri_1: str,
    vocab_uri_2: str,
    vocab_uri_3: str,
    vocab_uri_4: str,
    vocab_uri_5: str,
    vocab_uri_6: str,
    vocab_uri_7: str,
    vocab_uri_8: str,
    vocab_uri_9: str,
    generate_new_vocab: bool,
) -> NamedTuple('Outputs', [
    ('master_vocab_gcs_uri', str),
    ('experiment_name', str),
    ('experiment_run', str),
]):
    
    """
    combine layer dictionaires to master dictionary
    master dictionary passed to train job for layer vocabs
    """
    
    # import packages
    import os
    import logging
    import pickle as pkl
    import time
    import numpy as np
    
    from google.cloud import storage
    
    # setup clients
    storage_client = storage.Client()
    
    if generate_new_vocab:
        
        logging.info(f"Generating new vocab master file...")
        # ===================================================
        # Create list of all layer vocab dict uris
        # ===================================================

        vocab_dict_uris = [
            vocab_uri_1, vocab_uri_2, 
            vocab_uri_3, vocab_uri_4, 
            vocab_uri_5, vocab_uri_6, 
            vocab_uri_7, vocab_uri_8, 
            vocab_uri_9, 
        ]
        logging.info(f"count of vocab_dict_uris: {len(vocab_dict_uris)}")
        logging.info(f"vocab_dict_uris: {vocab_dict_uris}")

        # ===================================================
        # load pickled dicts
        # ===================================================

        loaded_pickle_list = []
        for i, pickled_dict in enumerate(vocab_dict_uris):

            with open(f"v{i}_vocab_pre_load", 'wb') as local_vocab_file:
                storage_client.download_blob_to_file(pickled_dict, local_vocab_file)

            with open(f"v{i}_vocab_pre_load", 'rb') as pickle_file:
                loaded_vocab_dict = pkl.load(pickle_file)

            loaded_pickle_list.append(loaded_vocab_dict)

        # ===================================================
        # create master vocab dict
        # ===================================================
        master_dict = {}
        for loaded_dict in loaded_pickle_list:
            master_dict.update(loaded_dict)

        # ===================================================
        # Upload master to GCS
        # ===================================================
        MASTER_VOCAB_LOCAL_FILE = f'vocab_dict.pkl'
        MASTER_VOCAB_GCS_OBJ = f'{experiment_name}/{experiment_run}/{MASTER_VOCAB_LOCAL_FILE}' # destination folder prefix and blob name

        # pickle
        filehandler = open(f'{MASTER_VOCAB_LOCAL_FILE}', 'wb')
        pkl.dump(master_dict, filehandler)
        filehandler.close()

        # upload to GCS
        bucket_client = storage_client.bucket(train_output_gcs_bucket)
        blob = bucket_client.blob(MASTER_VOCAB_GCS_OBJ)
        blob.upload_from_filename(MASTER_VOCAB_LOCAL_FILE)

        master_vocab_uri = f'gs://{train_output_gcs_bucket}/{MASTER_VOCAB_GCS_OBJ}'

        logging.info(f"File {MASTER_VOCAB_LOCAL_FILE} uploaded to {master_vocab_uri}")
        
    else:
        logging.info(f"Using existing vocab file...")
        master_vocab_uri = 'gs://two-tower-models/vocabs/vocab_dict.pkl'
        logging.info(f"Using vocab file: {master_vocab_uri}")
    
    return(
        master_vocab_uri,
        experiment_name,
        experiment_run
    )
