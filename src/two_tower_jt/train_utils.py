import os
import numpy as np
from typing import Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import logging

from google.cloud import storage

# ===================================================
# get accelerator_config
# ===================================================
def get_accelerator_config(
    key: str, 
    reduction_n: Optional[int],
    accelerator_per_machine: int = 1, 
    worker_n: int = 1,
    worker_machine_type: str = 'n1-highmem-16',
    reduction_machine_type: str = "n1-highcpu-16", 
    distribute: str = 'single',
):
    """
    returns GPU configuration for vertex training
    
    example:
        desired_config = get_accelerator_config(MY_CHOICE)
    """
    if key == "a100-40":
        WORKER_MACHINE_TYPE = 'a2-highgpu-1g'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = 'NVIDIA_TESLA_A100'
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        REDUCTION_SERVER_COUNT = reduction_n                                                 
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
        DISTRIBUTE_STRATEGY = distribute
    if key == "a100-80":
        WORKER_MACHINE_TYPE = 'a2-ultragpu-1g'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = 'NVIDIA_A100_80GB'
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        REDUCTION_SERVER_COUNT = reduction_n                                                 
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
        DISTRIBUTE_STRATEGY = distribute
    elif key == 't4':
        WORKER_MACHINE_TYPE = worker_machine_type          #'n1-standard-16'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = 'NVIDIA_TESLA_T4'               # NVIDIA_TESLA_V100
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        DISTRIBUTE_STRATEGY = distribute
        REDUCTION_SERVER_COUNT = reduction_n                                                   
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
    elif key == 'v100':
        WORKER_MACHINE_TYPE = worker_machine_type          #'n1-standard-16'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = 'NVIDIA_TESLA_V100'
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        DISTRIBUTE_STRATEGY = distribute
        REDUCTION_SERVER_COUNT = reduction_n                                                   
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
    elif key == "no_gpu":
        WORKER_MACHINE_TYPE = worker_machine_type          #'n2-highmem-32'|'n1-highmem-96'|'n2-highmem-92'
        REPLICA_COUNT = worker_n
        ACCELERATOR_TYPE = None
        PER_MACHINE_ACCELERATOR_COUNT = accelerator_per_machine
        DISTRIBUTE_STRATEGY = distribute
        REDUCTION_SERVER_COUNT = reduction_n                                                 
        REDUCTION_SERVER_MACHINE_TYPE = reduction_machine_type
    else:
        print(f"Incorrect key entry. Select from ['a100-40','a100-80', 't4', 'v100', 'no_gpu']")

    print(f"WORKER_MACHINE_TYPE            : {WORKER_MACHINE_TYPE}")
    print(f"REPLICA_COUNT                  : {REPLICA_COUNT}")
    print(f"ACCELERATOR_TYPE               : {ACCELERATOR_TYPE}")
    print(f"PER_MACHINE_ACCELERATOR_COUNT  : {PER_MACHINE_ACCELERATOR_COUNT}")
    print(f"DISTRIBUTE_STRATEGY            : {DISTRIBUTE_STRATEGY}")
    print(f"REDUCTION_SERVER_COUNT         : {REDUCTION_SERVER_COUNT}")
    print(f"REDUCTION_SERVER_MACHINE_TYPE  : {REDUCTION_SERVER_MACHINE_TYPE}")
    
    accelerator_dict = {
        "WORKER_MACHINE_TYPE": WORKER_MACHINE_TYPE,
        "REPLICA_COUNT": REPLICA_COUNT,
        "ACCELERATOR_TYPE": ACCELERATOR_TYPE,
        "PER_MACHINE_ACCELERATOR_COUNT": PER_MACHINE_ACCELERATOR_COUNT,
        'REDUCTION_SERVER_COUNT': REDUCTION_SERVER_COUNT,
        "REDUCTION_SERVER_MACHINE_TYPE": REDUCTION_SERVER_MACHINE_TYPE,
        "DISTRIBUTE_STRATEGY": DISTRIBUTE_STRATEGY,
    }
    return accelerator_dict

# ====================================================
# TensorBoard Callbacks
# ====================================================
def get_upload_logs_to_manged_tb_command(
    ttl_hrs,
    LOG_DIR,
    TB_RESOURCE_NAME,
    EXPERIMENT_NAME,
    oneshot="false"
):
    """
    Run this and copy/paste the command into terminal to have 
    upload the tensorboard logs from this machine to the managed tb instance
    Note that the log dir is at the granularity of the run to help select the proper
    timestamped run in Tensorboard
    You can also run this in one-shot mode after training is done 
    to upload all tb objects at once
    """
    return(f"""tb-gcp-uploader --tensorboard_resource_name={TB_RESOURCE_NAME} \
      --logdir={LOG_DIR} \
      --experiment_name={EXPERIMENT_NAME} \
      --one_shot={oneshot} \
      --event_file_inactive_secs={60*60*ttl_hrs}""")

# tensorboard callback
class UploadTBLogsBatchEnd(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        tb_resource_name: int
    ):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.tb_resource_name = tb_resource_name
    '''
    ecapsulates one-shot log uploader via a custom callback

    '''
    def on_epoch_end(self, epoch, logs=None):
        os.system(
            get_upload_logs_to_manged_tb_command(
                ttl_hrs = 5, 
                oneshot="true",
                LOG_DIR=log_dir,
                TB_RESOURCE_NAME=tb_resource_name,
                EXPERIMENT_NAME=experiment_name,
            )
        )
        
# ====================================================
# Helper functions
# ====================================================
# upload files to Google Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name, project_id):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name" (no 'gs://')
    # source_file_name = "local/path/to/file" (file to upload)
    # destination_blob_name = "folder/paths-to/storage-object-name"
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
    
def download_blob(project_id, bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )
    
def get_buckets_20(MAX_VAL):
    """ 
    creates discretization buckets of size 20
    """
    list_buckets = list(np.linspace(0, MAX_VAL, num=20))
    return(list_buckets)

def tf_if_null_return_zero(val):
    """
    > a trick to remove NANs post tf2.0
    > this function fills in nans to zeros - sometimes happens in embedding calcs.
    > this will clean the embedding inputs downstream
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
    ''' Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results

def get_train_strategy(distribute_arg):

    # Single Machine, single compute device
    if distribute_arg == 'single':
        if tf.config.list_physical_devices('GPU'): # TODO: replace with - tf.config.list_physical_devices('GPU') | tf.test.is_gpu_available()
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
    # used for interleave - takes tensors and returns a tf.dataset
    data = tf.data.TFRecordDataset(data)
    return data

# full_parse, get_train_strategy, _is_chief, get_arch_from_string, tf_if_null_return_zero, get_buckets_20, upload_blob