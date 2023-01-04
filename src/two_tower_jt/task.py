import os
import sys
# from absl import app
# from absl import flags
# from absl import logging

import argparse
import logging
import json
import time
import pickle as pkl
import numpy as np

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.python.client import device_lib

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
# import hypertune
import traceback
from google.cloud.aiplatform.training_utils import cloud_profiler

# ====================================================
# Args
# ====================================================
def parse_args():
    """
    Parses command line arguments
    
    type: int, float, str
          bool() converts empty strings to `False` and non-empty strings to `True`
          see more details here: https://docs.python.org/3/library/argparse.html#type
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, required=False)
    parser.add_argument('--train_dir', type=str, required=False)
    parser.add_argument('--train_dir_prefix', type=str, required=False)
    parser.add_argument('--valid_dir', type=str, required=False)
    parser.add_argument('--valid_dir_prefix', type=str, required=False)
    parser.add_argument('--candidate_file_dir', type=str, required=False)
    parser.add_argument('--candidate_files_prefix', type=str, required=False)
    parser.add_argument('--train_output_gcs_bucket', type=str, required=False)
    parser.add_argument('--experiment_name', type=str, required=False)
    parser.add_argument('--experiment_run', type=str, required=False)
    parser.add_argument('--num_epochs', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--embedding_dim', type=int, required=False)
    parser.add_argument('--projection_dim', type=int, required=False)
    parser.add_argument('--layer_sizes', type=str, required=False)
    parser.add_argument('--learning_rate', type=float, required=False)
    parser.add_argument('--valid_frequency', type=int, required=False)
    parser.add_argument('--valid_steps', type=int, required=False)
    parser.add_argument('--epoch_steps', type=int, required=False)
    parser.add_argument('--distribute', type=str, required=False)
    parser.add_argument('--model_version', type=str, required=False)
    parser.add_argument('--pipeline_version', type=str, required=False)
    parser.add_argument('--seed', type=int, required=False)
    parser.add_argument('--max_tokens', type=int, required=False)
    parser.add_argument('--tb_resource_name', type=str, required=False)
    parser.add_argument('--embed_frequency', type=int, required=False)
    parser.add_argument('--hist_frequency', type=int, required=False)
    parser.add_argument('--tf_gpu_thread_count', type=str, required=False)
    parser.add_argument('--block_length', type=int, required=False)
    parser.add_argument('--num_data_shards', type=int, required=False)
    parser.add_argument("--cache_train", action='store_true', help="include for True; ommit for False") #action=argparse.BooleanOptionalAction) # drop for False; included for True
    parser.add_argument("--evaluate_model", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--write_embeddings", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--profiler", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--set_jit", action='store_true', help="include for True; ommit for False")
    
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
import two_tower as tt       # import the model from the same module

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
    os.environ['TF_GPU_THREAD_MODE']='gpu_private'
    os.environ['TF_GPU_THREAD_COUNT']=f'{args.tf_gpu_thread_count}' # '1'
    os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
    
    # ====================================================
    # Set variables
    # ====================================================
    invoke_time = time.strftime("%Y%m%d-%H%M%S")
    # EXPERIMENT_NAME = args.experiment_name    
    # RUN_NAME = args.experiment_run
    OUTPUT_BUCKET = args.train_output_gcs_bucket
    LOG_DIR = f'gs://{OUTPUT_BUCKET}/{args.experiment_name}/{args.experiment_run}'
    
    logging.info(f'invoke_time: {invoke_time}')
    logging.info(f'EXPERIMENT_NAME: {args.experiment_name}')
    logging.info(f'RUN_NAME: {args.experiment_run}')
    logging.info(f'NUM_EPOCHS: {args.num_epochs}')
    logging.info(f'OUTPUT_BUCKET: {OUTPUT_BUCKET}')
    logging.info(f'LOG_DIR: {LOG_DIR}')
    logging.info(f'batch_size: {args.batch_size}')
    logging.info(f'train_dir: {args.train_dir}')
    logging.info(f'train_dir_prefix: {args.train_dir_prefix}')
    logging.info(f'valid_dir: {args.valid_dir}')
    logging.info(f'valid_dir_prefix: {args.valid_dir_prefix}')
    logging.info(f'embedding_dim: {args.embedding_dim}')
    logging.info(f'projection_dim: {args.projection_dim}')
    logging.info(f'learning_rate: {args.learning_rate}')
    logging.info(f'distribute: {args.distribute}')
    logging.info(f'model_version: {args.model_version}')
    logging.info(f'pipeline_version: {args.pipeline_version}')
    logging.info(f'max_tokens: {args.max_tokens}')
    logging.info(f'tb_resource_name: {args.tb_resource_name}')
    logging.info(f'valid_frequency: {args.valid_frequency}')
    logging.info(f'valid_steps: {args.valid_steps}')
    logging.info(f'epoch_steps: {args.epoch_steps}')
    logging.info(f'embed_frequency: {args.embed_frequency}')
    logging.info(f'hist_frequency: {args.hist_frequency}')
    logging.info(f'cache_train: {args.cache_train}')
    logging.info(f'evaluate_model: {args.evaluate_model}')
    logging.info(f'write_embeddings: {args.write_embeddings}')
    logging.info(f'profiler: {args.profiler}')
    logging.info(f'tf_gpu_thread_count: {args.tf_gpu_thread_count}')
    logging.info(f'set_jit: {args.set_jit}')
    logging.info(f'block_length: {args.block_length}')
    logging.info(f'num_data_shards: {args.num_data_shards}')
    
    # clients
    storage_client = storage.Client()
    
    vertex_ai.init(
        project=args.project,
        location='us-central1',
        experiment=args.experiment_name
    )
    
    # ====================================================
    # Set Device Strategy
    # ====================================================
    logging.info("Detecting devices....")
    # logging.info(f'Detected Devices {str(device_lib.list_local_devices())}')
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
    GLOBAL_BATCH_SIZE = int(args.batch_size) * int(NUM_REPLICAS)
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
    logging.info(f'Downloading vocab file...')
    
    os.system('gsutil cp gs://two-tower-models/vocabs/vocab_dict.pkl .')  # TODO - paramterize

    filehandler = open('vocab_dict.pkl', 'rb')
    vocab_dict = pkl.load(filehandler)
    filehandler.close()
    
    # ====================================================
    # Data loading
    # ====================================================
    # TODO - move to seperate py module in repo?
    logging.info(f'Preparing train, valid, and candidate tfrecords...\n')
    
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    
    # Disable intra-op parallelism to optimize for throughput instead of latency.
    options.threading.max_intra_op_parallelism = 1 # TODO 
    
    # ==============
    # Train data
    # ==============
    logging.info(f'Path to TRAIN files: gs://{args.train_dir}/{args.train_dir_prefix}')

    train_files = []
    for blob in storage_client.list_blobs(f'{args.train_dir}', prefix=f'{args.train_dir_prefix}'):
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

    train_dataset = train_dataset.interleave( # Parallelize data reading
        full_parse,
        cycle_length=tf.data.AUTOTUNE,
        block_length=args.block_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).repeat(
        args.num_epochs
    ).batch( #vectorize mapped function
        GLOBAL_BATCH_SIZE,
        drop_remainder=True,
    ).map(
        tt.parse_tfrecord, 
        num_parallel_calls=tf.data.AUTOTUNE
    # ).repeat(
    #     args.num_epochs
    # ).batch(
    #     GLOBAL_BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    ).with_options(
        options
    )
    
    if args.cache_train:
        logging.info("caching train_dataset in memory...")
        train_dataset.cache()
        logging.info("train_dataset should be cached in memory...")
        logging.info(f"train_dataset: {train_dataset}")
    
    # ==============
    # Valid data
    # ==============
    logging.info(f'Path to VALID files: gs://{args.valid_dir}/{args.valid_dir_prefix}')
    
    valid_files = []
    for blob in storage_client.list_blobs(f'{args.valid_dir}', prefix=f'{args.valid_dir_prefix}'):
        if '.tfrecords' in blob.name:
            valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    logging.info("Creating cached VALID dataset...")
    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files).prefetch(
        tf.data.AUTOTUNE,
    )

    valid_dataset = valid_dataset.interleave(
        full_parse,
        num_parallel_calls=tf.data.AUTOTUNE,
        cycle_length=tf.data.AUTOTUNE, 
        deterministic=False,
    ).map(tt.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE).with_options(options)

    valid_dataset = valid_dataset.cache() #1gb machine mem + 400 MB in candidate ds (src/two-tower.py)
    
    # ==============
    # candidate data
    # ==============
    logging.info(f'Path to CANDIDATE file(s): gs://{args.candidate_file_dir}/{args.candidate_files_prefix}')

    candidate_files = []
    for blob in storage_client.list_blobs(f"{args.candidate_file_dir}", prefix=f'{args.candidate_files_prefix}'):
        if '.tfrecords' in blob.name:
            candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    logging.info("Creating cached CANDIDATE dataset...")
    candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)
    
    parsed_candidate_dataset = candidate_dataset.interleave(
        # lambda x: tf.data.TFRecordDataset(x),
        full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    ).map(tt.parse_candidate_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE).with_options(options)

    parsed_candidate_dataset = parsed_candidate_dataset.cache() #400 MB on machine mem

    # ====================================================
    # Compile model
    # ====================================================
    logging.info('Setting model adapts and compiling model...')
    
    NUM_EPOCHS = args.num_epochs
    LAYER_SIZES = get_arch_from_string(args.layer_sizes)
    
    tf.config.optimizer.set_jit(args.set_jit) # Enable XLA.
    
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
    
    # ====================================================
    # callbacks
    # ====================================================
    
    log_dir = f"{LOG_DIR}/tb-logs-jt"
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR']
        logging.info(f'AIP_TENSORBOARD_LOG_DIR: {log_dir}')
        
    logging.info(f'log_dir for TensorBoard: {log_dir}')
    
#     backup_and_restore_callback = tf.keras.callbacks.experimental.BackupAndRestore(
#         backup_dir=os.environ['AIP_CHECKPOINT_DIR']
#     )
    

    if args.profiler:
        #TODO
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=args.hist_frequency, 
            write_graph=True,
            # embeddings_freq=args.embed_frequency,
            profile_batch=(20, 30),
        )
        logging.info(f'Tensorboard callback should profile batches...')
        
    else:
        # TODO
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=args.hist_frequency, 
            write_graph=True,
            # embeddings_freq=args.embed_frequency,
        )
        logging.info(f'Tensorboard callback NOT profiling batches...')

    # ====================================================
    # Train model
    # ====================================================
    
    # Initialize the profiler.
    logging.info('Initialize the profiler ...')
        
    try:
        cloud_profiler.init()
    except:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        print("*** Unexpected:", ex_type.__name__, ex_value)
        traceback.print_tb(ex_traceback, limit=10, file=sys.stdout)
    
    logging.info('Starting training loop...')
    start_time = time.time()
    
    layer_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        validation_freq=args.valid_frequency,
        epochs=NUM_EPOCHS,
        steps_per_epoch=args.epoch_steps,
        validation_steps=args.valid_steps, # 100,
        callbacks=[
            tensorboard_callback,
            # backup_and_restore_callback
        ], 
        verbose=1
    )

    end_time = time.time()
    val_keys = [v for v in layer_history.history.keys()]
    total_train_time = int((end_time - start_time) / 60)
    metrics_dict = {"total_train_time": total_train_time}
    logging.info(f"total_train_time: {total_train_time}")
    
    _ = [metrics_dict.update({key: layer_history.history[key][-1]}) for key in val_keys]
    
    if args.evaluate_model:
        logging.info(f"beginning model eval...")
        
        start_time = time.time()
        
        eval_dict = model.evaluate(valid_dataset, return_dict=True)
        
        end_time = time.time()
        
        total_eval_time = int((end_time - start_time) / 60)
        logging.info(f"total_eval_time: {total_eval_time}")
        logging.info(f"eval_dict: {eval_dict}")
        
        if task_type == 'chief':
            logging.info(f"Chief saving model eval dict...")
            filehandler = open('model_eval_dict.pkl', 'wb')
            pkl.dump(eval_dict, filehandler)
            filehandler.close()
            tt.upload_blob(f'{OUTPUT_BUCKET}', 'model_eval_dict.pkl', f'{args.experiment_name}/{args.experiment_run}/combined-model-eval/model_eval_dict.pkl')
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    # IF CHIEF, LOG to EXPERIMENT
    if task_type == 'chief':
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")
        
        with vertex_ai.start_run(
            args.experiment_run, 
            # tensorboard=args.tb_resource_name
        ) as my_run:
            
            logging.info(f"logging metrics...")
            my_run.log_metrics(metrics_dict)

            logging.info(f"logging metaparams...")
            my_run.log_params(
                {
                    "layers": str(LAYER_SIZES),
                    "num_epochs": NUM_EPOCHS,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "valid_freq": args.valid_frequency,
                    "embed_freq": args.embed_frequency,
                    "hist_freq": args.hist_frequency,
                    # "xxx": args.xxxx,
                }
            )

            vertex_ai.end_run()
            logging.info(f"EXPERIMENT RUN: {args.experiment_run} has ended")

    # ====================================================
    # Save model
    # ====================================================
    MODEL_DIR_GCS_URI = f'{LOG_DIR}/model-dir'
    logging.info(f"Saving models to {MODEL_DIR_GCS_URI}")
    
    # save model from primary node in multiworker
    if task_type == 'chief':
        #query tower
        query_model_dir = f"{MODEL_DIR_GCS_URI}/query_model"
        tf.saved_model.save(model.query_tower, export_dir=query_model_dir)
        logging.info(f'Saved chief query model to {query_model_dir}')
        # candidate tower
        candidate_model_dir = f"{MODEL_DIR_GCS_URI}/candidate_model"
        tf.saved_model.save(model.candidate_tower, export_dir=candidate_model_dir)
        logging.info(f'Saved chief candidate model to {candidate_model_dir}')
        
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

    logging.info('All done - models saved') #all done

    # ====================================================
    # Save embeddings
    # ====================================================
    
    if args.write_embeddings:
        # TODO: 
        logging.info('Saving candidate embeddings...')
        Local_Candidate_Embedding_Index = 'candidate_embeddings.json'
    
        candidate_embeddings = parsed_candidate_dataset.batch(10000).map(lambda x: [x['track_uri_can'], tf_if_null_return_zero(model.candidate_tower(x))])
    
        # Save to the required format
        for batch in candidate_embeddings:
            songs, embeddings = batch
            with open(f"{Local_Candidate_Embedding_Index}", 'a') as f:
                for song, emb in zip(songs.numpy(), embeddings.numpy()):
                    f.write('{"id":"' + str(song) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}')
                    f.write("\n")

        if task_type == 'chief':
            tt.upload_blob(
                f'{OUTPUT_BUCKET}', 
                f'{Local_Candidate_Embedding_Index}', 
                f'{args.experiment_name}/{args.experiment_run}/candidates/{Local_Candidate_Embedding_Index}'
            )
    
        logging.info(f"Saved {Local_Candidate_Embedding_Index} to {LOG_DIR}/candidates/{Local_Candidate_Embedding_Index}")


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
