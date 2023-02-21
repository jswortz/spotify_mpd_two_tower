
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
    data_dir_bucket_name: str,
    data_dir_path_prefix: str,
    vocab_path_prefix: str,
    master_dict_path_prefix: str,
    vocab_uri_1: str,
    vocab_uri_2: str,
    vocab_uri_3: str,
    vocab_uri_4: str,
    vocab_uri_5: str,
    vocab_uri_6: str,
    vocab_uri_7: str,
    vocab_uri_8: str,
    vocab_uri_9: str,
    # vocab_uri_10: str,
    # vocab_uri_11: str,
) -> NamedTuple('Outputs', [
    ('master_vocab_gcs_uri', str),
    # ('feature_name': str),
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

    # ===================================================
    # Create list of all layer vocab dict uris
    # ===================================================
    
    vocab_dict_uris = [
        vocab_uri_1, vocab_uri_2, 
        vocab_uri_3, vocab_uri_4, 
        vocab_uri_5, vocab_uri_6, 
        vocab_uri_7, vocab_uri_8, 
        vocab_uri_9, 
        # vocab_uri_10, 
        # vocab_uri_11,
    ]
        
    # vocab_dict_uris = []
    # for blob in storage_client.list_blobs(f'{data_dir_bucket_name}', prefix=f'{vocab_path_prefix}'):
    #     if '.pkl' in blob.name:
    #         vocab_dict_uris.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
        
    # logging.info(f"vocab_dict_uris[0]: {vocab_dict_uris[0]}")
    
    # skip folder path prefix
    # vocab_dict_uris = vocab_dict_uris[1:]
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
    MASTER_VOCAB_LOCAL_FILE = f'{version}_master_vocab_dict.pkl'
    MASTER_VOCAB_GCS_OBJ = f'{master_dict_path_prefix}/{MASTER_VOCAB_LOCAL_FILE}' # destination folder prefix and blob name
    
    # pickle
    filehandler = open(f'{MASTER_VOCAB_LOCAL_FILE}', 'wb')
    pkl.dump(master_dict, filehandler)
    filehandler.close()
    
    # upload to GCS
    bucket_client = storage_client.bucket(data_dir_bucket_name)
    blob = bucket_client.blob(MASTER_VOCAB_GCS_OBJ)
    blob.upload_from_filename(MASTER_VOCAB_LOCAL_FILE)
    
    master_vocab_uri = f'gs://{data_dir_bucket_name}/{MASTER_VOCAB_GCS_OBJ}'
    
    logging.info(f"File {MASTER_VOCAB_LOCAL_FILE} uploaded to {master_vocab_uri}")
    
    return(
        master_vocab_uri,
    )
