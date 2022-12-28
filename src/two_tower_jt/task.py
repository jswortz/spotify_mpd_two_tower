import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'

# from absl import app
# from absl import flags
# from absl import logging

import logging
import json

import tensorflow as tf
import logging
import time
import pickle as pkl
import numpy as np

import tensorflow_recommenders as tfrs
from tensorflow.python.client import device_lib

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
# import hypertune
# from google.cloud.aiplatform.training_utils import cloud_profiler

import two_tower as tt #import the model from the same module

# ====================================================
# Args
# ====================================================
# FLAGS = flags.FLAGS
# flags.DEFINE_string("train_dir", 'spotify-data-regimes', "bucket where tfrecords live")
# flags.DEFINE_string("train_dir_prefix", 'v10/train_v9/','path to training data in train_dir')
# flags.DEFINE_string("valid_dir_prefix", 'v10/valid_v9/','path to validation data in train_dir')
# flags.DEFINE_string("EXPERIMENT_NAME", None,'vertex ai experiment name')

# flags.DEFINE_string("BUCKET", 'two-tower-models','name of bucket for assets')
# flags.DEFINE_integer("SEED", 41781897, "random seed")

# flags.DEFINE_float("LR", 0.1, "Learning Rate")
# flags.DEFINE_bool("DROPOUT", False, "Use Dropout - T/F bool type")
# flags.DEFINE_float("DROPOUT_RATE", 0.4, "Dropout rate only works with DROPOUT=True")
# flags.DEFINE_integer("EMBEDDING_DIM", 128, "Embedding dimension")
# flags.DEFINE_string("ARCH", None, "deep architecture, expressed as a list of ints in string format - will be parsed into list")
# flags.DEFINE_integer("NUM_EPOCHS", None, "Number of epochs")
# flags.DEFINE_integer("BATCH_SIZE", None, "batch size")
# flags.DEFINE_integer("MAX_TOKENS", 20000, "Max embeddings for query and last_n tracks")

# flags.DEFINE_string("PROJECT_ID", "hybrid-vertex", "Project ID")
# flags.DEFINE_string("LOCATION", "us-central1", "GCP Location")
                    
# flags.DEFINE_string("TB_RESOURCE_NAME", 'projects/934903580331/locations/us-central1/tensorboards/7336372589079560192', "Vertex TB resource")

def parse_args():
    """
    Parses command line arguments
    
    type: int, float, str
          bool() converts empty strings to `False` and non-empty strings to `True`
          see more details here: https://docs.python.org/3/library/argparse.html#type
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=False)
    parser.add_argument('--train_dir', required=False)
    parser.add_argument('--train_dir_prefix', required=False)
    parser.add_argument('--valid_dir', required=False)
    parser.add_argument('--valid_dir_prefix', required=False)
    parser.add_argument('--candidate_file_dir', required=False)
    parser.add_argument('--candidate_files_prefix', required=False)
    parser.add_argument('--train_output_gcs_bucket', required=False)
    parser.add_argument('--experiment_name', required=False)
    parser.add_argument('--experiment_run', required=False)
    parser.add_argument('--num_epochs', required=False)
    parser.add_argument('--batch_size', required=False)
    parser.add_argument('--embedding_dim', required=False)
    parser.add_argument('--projection_dim', required=False)
    parser.add_argument('--layer_sizes', required=False)
    parser.add_argument('--learning_rate', required=False)
    # parser.add_argument('--valid_frequency', required=False)
    parser.add_argument('--distribute', required=False)
    parser.add_argument('--model_version', required=False)
    parser.add_argument('--pipeline_version', required=False)
    parser.add_argument('--seed', required=False)
    parser.add_argument('--max_tokens', required=False)
    parser.add_argument('--tb_resource_name', required=False)
    
    return parser.parse_args()

# ====================================================
# Helper functions
# ====================================================

def tf_if_null_return_zero(val):
    """
    this function fills in nans to zeros - sometimes happens in embedding calcs.
    this will clean the embedding inputs downstream
    """
    return(tf.clip_by_value(val, -1e12, 1e12)) # a trick to remove NANs post tf2.0

def get_arch_from_string(arch_string):
    q = arch_string.replace(']', '')
    q = q.replace('[', '')
    q = q.replace(" ", "")
    return [int(x) for x in q.split(',')]

def _is_chief(task_type, task_id): 
    ''' Check for primary if multiworker training
    '''
    if task_type == 'chief':
        results = 'chief'
    else:
        results = None
    return results

# ====================================================
# Main
# ====================================================

import train_config as cfg

def main(args):
    
    # limiting GPU growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f'detected: {len(gpus)} GPUs')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logging.info(e)
                
    # tf.debugging.set_log_device_placement(True) # logs all tf ops and their device placement;
    # TF_GPU_THREAD_MODE='gpu_private'
    os.environ['TF_GPU_THREAD_MODE']='gpu_private'
    os.environ['TF_GPU_THREAD_COUNT']='1'
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    
    # ====================================================
    # Set variables
    # ====================================================
    invoke_time = time.strftime("%Y%m%d-%H%M%S")
    EXPERIMENT_NAME = args.experiment_name    
    RUN_NAME = args.experiment_run
    NUM_EPOCHS = args.num_epochs
    
    LAYER_SIZES=get_arch_from_string(args.layer_sizes)
    OUTPUT_BUCKET = args.train_output_gcs_bucket
    LOG_DIR = f'gs://{OUTPUT_BUCKET}/{EXPERIMENT_NAME}/{RUN_NAME}'

    batch_size = args.batch_size
    train_dir = args.train_dir
    train_dir_prefix = args.train_dir_prefix

    valid_dir = args.train_dir
    valid_dir_prefix = args.valid_dir_prefix
    
    logging.info(f'invoke_time: {invoke_time}')
    logging.info(f'EXPERIMENT_NAME: {EXPERIMENT_NAME}')
    logging.info(f'RUN_NAME: {RUN_NAME}')
    logging.info(f'NUM_EPOCHS: {NUM_EPOCHS}')
    logging.info(f'OUTPUT_BUCKET: {OUTPUT_BUCKET}')
    logging.info(f'LOG_DIR: {LOG_DIR}')
    logging.info(f'batch_size: {batch_size}')
    logging.info(f'train_dir: {train_dir}')
    logging.info(f'train_dir_prefix: {train_dir_prefix}')
    logging.info(f'valid_dir: {valid_dir}')
    logging.info(f'valid_dir_prefix: {valid_dir_prefix}')
    

    # clients
    storage_client = storage.Client()
    
    vertex_ai.init(
        project=args.project,
        location='us-central1',
        experiment=EXPERIMENT_NAME
    )
    
    # ====================================================
    # Set Device Strategy
    # ====================================================
    logging.info("Detecting devices....")
    logging.info(f'Detected Devices {str(device_lib.list_local_devices())}')
    logging.info("Setting device strategy...")
    
    # Single Machine, single compute device
    if args.distribute == 'single':
        if tf.config.list_physical_devices('GPU'): # TODO: replace with - tf.config.list_physical_devices('GPU') | tf.test.is_gpu_available()
            strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        logging.info("Single device training")  
    # Single Machine, multiple compute device
    elif args.distribute == 'mirrored':
        strategy = tf.distribute.MirroredStrategy()
        logging.info("Mirrored Strategy distributed training")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Multi Machine, multiple compute device
    elif args.distribute == 'multiworker':
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        logging.info("Multi-worker Strategy distributed training")
        logging.info('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))
    # Single Machine, multiple TPU devices
    elif args.distribute == 'tpu':
        cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
        tf.config.experimental_connect_to_cluster(cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
        strategy = tf.distribute.TPUStrategy(cluster_resolver)
        logging.info("All devices: ", tf.config.list_logical_devices('TPU'))

    logging.info(f'TF training strategy = {strategy}')
    
    NUM_REPLICAS = strategy.num_replicas_in_sync
    logging.info(f'num_replicas_in_sync = {NUM_REPLICAS}')
    
    # Here the batch size scales up by number of workers since
    # `tf.data.Dataset.batch` expects the global batch size.
    GLOBAL_BATCH_SIZE = batch_size * NUM_REPLICAS
    logging.info(f'GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}')

    # TODO: Determine type and task of the machine from the strategy cluster resolver
    logging.info(f'Setting task_type and task_id...')
    # task_type, task_id = (
    #     strategy.cluster_resolver.task_type,
    #     strategy.cluster_resolver.task_id
    # )
    if args.distribute == 'multiworker':
        task_type, task_id = (
            strategy.cluster_resolver.task_type,
            strategy.cluster_resolver.task_id
        )
    else:
        task_type, task_id = 'chief', None
    
    logging.info(f'task_type = {task_type}')
    logging.info(f'task_id = {task_id}')
    
    # ====================================================
    # Vocab Files
    # ====================================================
    
    # TODO: paramterize
    
    os.system('gsutil cp gs://two-tower-models/vocabs/vocab_dict.pkl .')  # TODO - paramterize

    filehandler = open('vocab_dict.pkl', 'rb')
    vocab_dict = pkl.load(filehandler)
    filehandler.close()
    
    
    # ====================================================
    # Data loading
    # ====================================================


    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
    
    logging.info(f'Path to TRAIN files: gs://{train_dir}/{train_dir_prefix}')

    train_files = []
    for blob in storage_client.list_blobs(f'{train_dir}', prefix=f'{train_dir_prefix}'):
        if '.tfrecords' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    def full_parse(data):
        # used for interleave - takes tensors and returns a tf.dataset
        data = tf.data.TFRecordDataset(data)
        return data
    
    logging.info("Creating TRAIN dataset...")

    train_dataset = tf.data.Dataset.from_tensor_slices(train_files).prefetch(
        tf.data.AUTOTUNE,
    )

    train_dataset = train_dataset.interleave(
        full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    ).map(tt.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE,).batch(
        batch_size 
    ).prefetch(
        tf.data.AUTOTUNE,
    ).with_options(options)
    
    # valid data
    valid_files = []
    for blob in storage_client.list_blobs(f'{valid_dir}', prefix=f'{valid_dir_prefix}'):
        if '.tfrecords' in blob.name:
            valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    logging.info("Creating VALID dataset...")

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files).prefetch(
        tf.data.AUTOTUNE,
    )

    valid_dataset = valid_dataset.interleave(
        full_parse,
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE, 
        deterministic=False,
    ).map(tt.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(
        tf.data.AUTOTUNE,
    ).with_options(options)

    valid_dataset = valid_dataset.cache() #1gb machine mem + 400 MB in candidate ds (src/two-tower.py)
    
    # candidate data
    CANDIDATE_PREFIX = 'jtv10/candidates' # TODO - paramterize

    candidate_files = []
    for blob in client.list_blobs(f"{train_dir}", prefix=f'{CANDIDATE_PREFIX}'):
        if '.tfrecords' in blob.name:
            candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    #generate the candidate dataset

    candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)
    parsed_candidate_dataset = candidate_dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).map(
        tt.parse_candidate_tfrecord_fn,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).with_options(options)

    parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem

    # ====================================================
    # Compile model
    # ====================================================
    logging.info('Setting model adapts and compiling the model')
    
    # Wrap variable creation within strategy scope
    with strategy.scope():

        model = tt.TheTwoTowers(
            LAYER_SIZES, 
            vocab_dict, 
            parsed_candidate_dataset,
            # max_padding_len=args.max_padding
        )
            
        model.compile(optimizer=tf.keras.optimizers.Adagrad(args.learning_rate))
    
    logging.info('model compiled...')
        
    tf.random.set_seed(args.seed)
    
    def get_upload_logs_to_manged_tb_command(ttl_hrs, oneshot="false"):
        """
        Run this and copy/paste the command into terminal to have 
        upload the tensorboard logs from this machine to the managed tb instance
        Note that the log dir is at the granularity of the run to help select the proper
        timestamped run in Tensorboard
        You can also run this in one-shot mode after training is done 
        to upload all tb objects at once
        """
        return(f"""tb-gcp-uploader --tensorboard_resource_name={args.tb_resource_name} \
          --logdir={LOG_DIR}/tb-logs \
          --experiment_name={EXPERIMENT_NAME} \
          --one_shot={oneshot} \
          --event_file_inactive_secs={60*60*ttl_hrs}""")
    
    # we are going to ecapsulate this one-shot log uploader via a custom callback:

    class UploadTBLogsBatchEnd(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            os.system(get_upload_logs_to_manged_tb_command(ttl_hrs = 5, oneshot="true"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=f"{LOG_DIR}/tb-logs",
            histogram_freq=0, 
            write_graph=True, 
            # profile_batch=(20,50) #run profiler on steps 20-40 - enable this line if you want to run profiler from the utils/ notebook
        )
    
    logging.info(f'TensorBoard logdir: {LOG_DIR}/tb-logs')


    # ====================================================
    # Train model
    # ====================================================
    
    cloud_profiler.init()
    
    logging.info('Starting training loop...')
    start_time = time.time()
    
    layer_history = model.fit(
        train_dataset.unbatch().batch(batch_size),
        validation_data=valid_dataset,
        validation_freq=3,
        epochs=NUM_EPOCHS,
        # steps_per_epoch=2, #use this for development to run just a few steps
        validation_steps = 100,
        callbacks=[
            tensorboard_callback,
            UploadTBLogsBatchEnd()
        ], #the tensorboard will be automatically associated with the experiment and log subsequent runs with this callback
        verbose=1
    )

    end_time = time.time()
    val_keys = [v for v in layer_history.history.keys()]
    runtime_mins = int((end_time - start_time) / 60)
    metrics_dict = {"runtime_mins": runtime_mins}
    logging.info(f"runtime_mins: {runtime_mins}")
    
    _ = [metrics_dict.update({key: layer_history.history[key][-1]}) for key in val_keys]
    
    # IF CHIEF, LOG to EXPERIMENT
    # if _is_chief(task_type, task_id):
    if task_type == 'chief':
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")
        
        with vertex_ai.start_run(RUN_NAME, tensorboard=args.tb_resource_name) as my_run:
            
            logging.info(f"logging metrics...")
            my_run.log_metrics(metrics_dict)

            logging.info(f"logging metaparams...")
            my_run.log_params(
                {
                    "layers": str(LAYER_SIZES),
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": batch_size,
                }
            )

            vertex_ai.end_run()
            logging.info(f"EXPERIMENT RUN: {RUN_NAME} has ended")

    # ====================================================
    # Save model
    # ====================================================
    
    MODEL_DIR_GCS_URI = f'{LOG_DIR}/model-dir'
    logging.info(f"Saving models to {MODEL_DIR_GCS_URI}")
    
    # save model from primary node in multiworker
    # if _is_chief(task_type, task_id):
    if task_type == 'chief':
        tf.saved_model.save(model.query_tower, export_dir=MODEL_DIR_GCS_URI + "/query_model")
        logging.info(f'Saved chief query model to {MODEL_DIR_GCS_URI}/query_model')
        tf.saved_model.save(model.candidate_tower, export_dir=MODEL_DIR_GCS_URI + "/candidate_model")
        logging.info(f'Saved chief candidate model to {MODEL_DIR_GCS_URI}/candidate_model')
    else:
        worker_dir_query = MODEL_DIR_GCS_URI + '/workertemp_query_/' + str(task_id)
        tf.io.gfile.makedirs(worker_dir_query)
        tf.saved_model.save(model.query_tower, worker_dir_query)
        logging.info(f'Saved worker: {task_id} query model to {worker_dir_query}')

        worker_dir_can = MODEL_DIR_GCS_URI + '/workertemp_can_/' + str(task_id)
        tf.io.gfile.makedirs(worker_dir_can)
        tf.saved_model.save(model.candidate_tower, worker_dir_can)
        logging.info(f'Saved worker: {task_id} candidate model to {worker_dir_can}')

    # if not _is_chief(task_type, task_id):
    if task_type != 'chief':
        tf.io.gfile.rmtree(worker_dir_can)
        tf.io.gfile.rmtree(worker_dir_query)

    logging.info('All done - model saved') #all done
    
    # tf.saved_model.save(model.query_tower, export_dir=MODEL_DIR_GCS_URI + "/query_model")
    # tf.saved_model.save(model.candidate_tower, export_dir=MODEL_DIR_GCS_URI + "/candidate_model")

    # ====================================================
    # Save embeddings
    # ====================================================

    candidate_embeddings = parsed_candidate_dataset.batch(10000).map(lambda x: [x['track_uri_can'], tf_if_null_return_zero(model.candidate_tower(x))])

    logging.info('Saving candidate embeddings...')
    
    # Save to the required format
    # make sure you start out with a clean empty file for the append write
    os.system('rm candidate_embeddings.json > /dev/null')
    os.system('touch candidate_embeddings.json')
    for batch in candidate_embeddings:
        songs, embeddings = batch
        with open("candidate_embeddings.json", 'a') as f:
            for song, emb in zip(songs.numpy(), embeddings.numpy()):
                f.write('{"id":"' + str(song) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}')
                f.write("\n")
    
    tt.upload_blob(
        f'{OUTPUT_BUCKET}', 
        'candidate_embeddings.json', 
        f'{EXPERIMENT_NAME}/{RUN_NAME}/candidates/candidate_embeddings.json'
    )


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        level=logging.INFO, 
        datefmt='%d-%m-%y %H:%M:%S',
        stream=sys.stdout
    )

    parsed_args = parse_args()

    logging.info('Args: %s', parsed_args)
    start_time = time.time()
    logging.info('Starting jobs main() script')

    main(parsed_args)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info('Training completed. Elapsed time: %s', elapsed_time )
