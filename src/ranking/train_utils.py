import os
import numpy as np
import logging

import tensorflow as tf

from google.cloud import storage

# ====================================================
# Helper functions
# ====================================================

# set project # env var
project_number = os.environ["CLOUD_ML_PROJECT_ID"]

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """
    Uploads a file to the bucket.
    """
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
    list_buckets = list(
        np.linspace(0, MAX_VAL, num=20)
    )
    return(list_buckets)

def tf_if_null_return_zero(val):
    """
    > a trick to remove NANs post tf2.0
    > this function fills in nans to zeros
    > this will clean the embedding inputs
    """
    return(
        tf.clip_by_value(val, -1e12, 1e12)
    )

def get_arch_from_string(arch_string):
    
    q = arch_string.replace(']', '')
    q = q.replace('[', '')
    q = q.replace(" ", "")
    
    return [
        int(x) for x in q.split(',')
    ]

def _is_chief(task_type, task_id): 
    ''' 
    Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results

def get_train_strategy(distribute_arg):

    # Single Machine, single compute device
    if distribute_arg == 'single':
        if tf.config.list_physical_devices('GPU'):
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")  
    
    # Single Machine, multiple compute device
    elif distribute_arg == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Multi Machine, multiple compute device
    elif distribute_arg == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    
    # Single Machine, multiple TPU devices
    elif distribute_arg == 'tpu':
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    return strategy

# data loading and parsing
def full_parse(data):
    '''
    Used for interleave
    takes tensors as inputs and returns a tf.dataset
    '''
    data = tf.data.TFRecordDataset(data)
    return data

# full_parse, get_train_strategy, _is_chief, get_arch_from_string, tf_if_null_return_zero, get_buckets_20, upload_blob