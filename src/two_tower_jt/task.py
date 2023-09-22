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
import random
import string

import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.python.client import device_lib

from google.cloud import aiplatform as vertex_ai
from google.cloud import storage
# import hypertune
import traceback
from google.cloud.aiplatform.training_utils import cloud_profiler

# import modules
# import train_config as cfg
# import train_utils
# import feature_sets
# import two_tower as tt

# new fix to train image + ENTRY CMD 
from . import train_config as cfg
from . import feature_sets
from . import train_utils
from . import two_tower as tt


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
    parser.add_argument('--update_frequency', required=False)
    parser.add_argument('--tf_gpu_thread_count', type=str, required=False)
    parser.add_argument('--block_length', type=int, required=False)
    parser.add_argument('--num_data_shards', type=int, required=False)
    parser.add_argument("--cache_train", action='store_true', help="include for True; ommit for False") # drop for False; included for True
    parser.add_argument("--evaluate_model", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--write_embeddings", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--profiler", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--set_jit", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--use_cross_layer", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--use_dropout", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--compute_batch_metrics", action='store_true', help="include for True; ommit for False")
    parser.add_argument("--new_vocab", action='store_true', help="include for True; ommit for False")
    parser.add_argument('--chkpt_freq', required=False) # type=int | TODO: value could be int or string
    parser.add_argument('--dropout_rate', type=float, required=False)
    
    return parser.parse_args()

# ====================================================
# Main
# ====================================================

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
    logging.info(f'chkpt_freq: {args.chkpt_freq}')
    logging.info(f'compute_batch_metrics: {args.compute_batch_metrics}')
    logging.info(f'use_cross_layer: {args.use_cross_layer}')
    logging.info(f'use_dropout: {args.use_dropout}')
    logging.info(f'dropout_rate: {args.dropout_rate}')
    logging.info(f'new_vocab: {args.new_vocab}')
    
    
    project_number = os.environ["CLOUD_ML_PROJECT_ID"]
    
    # clients
    storage_client = storage.Client(project=project_number)
    
    vertex_ai.init(
        project=project_number,
        location='us-central1',
        experiment=args.experiment_name
    )
    
    # ====================================================
    # Set Device Strategy
    # ====================================================
    logging.info("Detecting devices....")
    # logging.info(f'Detected Devices {str(device_lib.list_local_devices())}')
    
    logging.info("Setting device strategy...")
    
    strategy = train_utils.get_train_strategy(distribute_arg=args.distribute)
    logging.info(f'TF training strategy = {strategy}')
    
    NUM_REPLICAS = strategy.num_replicas_in_sync
    logging.info(f'num_replicas_in_sync = {NUM_REPLICAS}')
    
    # Here the batch size scales up by number of workers since
    # `tf.data.Dataset.batch` expects the global batch size.
    GLOBAL_BATCH_SIZE = int(args.batch_size) * int(NUM_REPLICAS)
    logging.info(f'GLOBAL_BATCH_SIZE = {GLOBAL_BATCH_SIZE}')

    # type and task of machine from strategy
    logging.info(f'Setting task_type and task_id...')
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

    LOCAL_VOCAB_FILENAME = 'vocab_dict.pkl'
    EXISTING_VOCAB_FILE = f'gs://{args.train_output_gcs_bucket}/{args.experiment_name}/{args.experiment_run}/{LOCAL_VOCAB_FILENAME}'
    logging.info(f'Downloading vocab file from: {EXISTING_VOCAB_FILE}...')
    
    train_utils.download_blob(
        project_id = args.project,
        bucket_name = args.train_output_gcs_bucket, 
        source_blob_name = f'{args.experiment_name}/{args.experiment_run}/{LOCAL_VOCAB_FILENAME}', 
        destination_file_name= LOCAL_VOCAB_FILENAME
    )

    logging.info(f"Downloaded vocab from: {EXISTING_VOCAB_FILE}\n")

    filehandler = open(f"{LOCAL_VOCAB_FILENAME}", 'rb')
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
    logging.info("Creating TRAIN dataset...")
    logging.info(f'Path to TRAIN files: gs://{args.train_dir}/{args.train_dir_prefix}')

    train_files = []
    for blob in storage_client.list_blobs(f'{args.train_dir}', prefix=f'{args.train_dir_prefix}'):
        if '.tfrecords' in blob.name:
            train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            # train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "/gcs/"))
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files).prefetch(
        tf.data.AUTOTUNE,
    )

    train_dataset = train_dataset.interleave( # Parallelize data reading
        train_utils.full_parse,
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
        feature_sets.parse_tfrecord, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(
        tf.data.AUTOTUNE # GLOBAL_BATCH_SIZE*3 # tf.data.AUTOTUNE
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
    logging.info("Creating cached VALID dataset...")
    logging.info(f'Path to VALID files: gs://{args.valid_dir}/{args.valid_dir_prefix}')
    
    valid_files = []
    for blob in storage_client.list_blobs(f'{args.valid_dir}', prefix=f'{args.valid_dir_prefix}'):
        if '.tfrecords' in blob.name:
            valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            # valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "/gcs/"))

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files).prefetch(
        tf.data.AUTOTUNE,
    )

    valid_dataset = valid_dataset.interleave( # Parallelize data reading
        train_utils.full_parse,
        num_parallel_calls=tf.data.AUTOTUNE,
        block_length=args.block_length,
        cycle_length=tf.data.AUTOTUNE, 
        deterministic=False,
    ).batch(
        GLOBAL_BATCH_SIZE
    ).map(
        feature_sets.parse_tfrecord, 
        num_parallel_calls=tf.data.AUTOTUNE
    # ).batch(
    #     GLOBAL_BATCH_SIZE
    ).prefetch(
        tf.data.AUTOTUNE
    ).with_options(
        options
    )
    valid_dataset = valid_dataset.cache()
    
    # ==============
    # candidate data
    # ==============
    logging.info("Creating cached CANDIDATE dataset...")
    logging.info(f'Path to CANDIDATE file(s): gs://{args.candidate_file_dir}/{args.candidate_files_prefix}')

    candidate_files = []
    for blob in storage_client.list_blobs(f"{args.candidate_file_dir}", prefix=f'{args.candidate_files_prefix}'):
        if '.tfrecords' in blob.name:
            candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))
            # candidate_files.append(blob.public_url.replace("https://storage.googleapis.com/", "/gcs/"))

    candidate_dataset = tf.data.Dataset.from_tensor_slices(candidate_files)
    
    parsed_candidate_dataset = candidate_dataset.interleave(
        train_utils.full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    # ).batch(
    #     GLOBAL_BATCH_SIZE
    ).map(
        feature_sets.parse_candidate_tfrecord_fn, 
        num_parallel_calls=tf.data.AUTOTUNE
    ).with_options(
        options
    )
    parsed_candidate_dataset = parsed_candidate_dataset.cache()

    # ====================================================
    # Compile model
    # ====================================================
    logging.info('Setting model adapts and compiling model...')
    
    LAYER_SIZES = train_utils.get_arch_from_string(args.layer_sizes)
    
    tf.config.optimizer.set_jit(args.set_jit) # Enable XLA.
    
    # Wrap variable creation within strategy scope
    with strategy.scope():

        model = tt.TheTwoTowers(
            layer_sizes=LAYER_SIZES, 
            vocab_dict=vocab_dict, 
            parsed_candidate_dataset=parsed_candidate_dataset,
            embedding_dim=args.embedding_dim,
            projection_dim=args.projection_dim,
            seed=args.seed,
            use_cross_layer=args.use_cross_layer,
            use_dropout=args.use_dropout,
            dropout_rate=args.dropout_rate,
            # max_playlist_length=MAX_PLAYLIST_LENGTH,
            max_tokens=args.max_tokens,
            compute_batch_metrics=args.compute_batch_metrics,
        )
            
        model.compile(optimizer=tf.keras.optimizers.Adagrad(args.learning_rate))
    
    logging.info('model compiled...')
        
    tf.random.set_seed(args.seed)
    
    # ====================================================
    # callbacks-v2
    # ====================================================
    
    log_dir = f"{LOG_DIR}/logs" # logs-jt
    if 'AIP_TENSORBOARD_LOG_DIR' in os.environ:
        log_dir=os.environ['AIP_TENSORBOARD_LOG_DIR']
        logging.info(f'AIP_TENSORBOARD_LOG_DIR: {log_dir}')
        
    logging.info(f'TensorBoard log_dir: {log_dir}')
    
    checkpoint_dir=os.environ['AIP_CHECKPOINT_DIR']
    logging.info(f'Saving model checkpoints to {checkpoint_dir}')
    
    # model checkpoints - ModelCheckpoint | BackupAndRestore
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + "/cp-{epoch:03d}-loss={loss:.2f}.ckpt", # checkpoint_dir
        save_weights_only=True,
        save_best_only=True,
        monitor='total_loss',
        mode='min',
        save_freq='epoch', #args.chkpt_freq,
        verbose=1,
    )

    if args.profiler:
        #TODO
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=args.hist_frequency, 
            write_graph=True,
            profile_batch=(25, 30),
            update_freq=args.update_frequency,  # 'epoch'
            embeddings_freq=args.embed_frequency,
            embeddings_metadata=log_dir + "/embs/metadata.tsv"
        )
        logging.info(f'Tensorboard callback should profile batches...')
        
    else:
        # TODO
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=args.hist_frequency, 
            write_graph=True,
            update_freq=args.update_frequency,  # 'epoch'
            embeddings_freq=args.embed_frequency,
            embeddings_metadata=log_dir + "/embs/metadata.tsv"
        )
        logging.info(f'Tensorboard callback NOT profiling batches...')

    # ====================================================
    # Train model
    # ====================================================
    
    if args.profiler:
        # Initialize profiler
        logging.info('Initializing profiler ...')

        # try:
        #     cloud_profiler.init()
        # except:
        #     ex_type, ex_value, ex_traceback = sys.exc_info()
        #     print("*** Unexpected:", ex_type.__name__, ex_value)
        #     traceback.print_tb(ex_traceback, limit=10, file=sys.stdout)

        logging.info('The profiler initiated...')
        
    # training loop
    logging.info('Starting training loop...')
    
    start_time = time.time()
    layer_history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        validation_freq=args.valid_frequency,
        epochs=args.num_epochs,
        steps_per_epoch=args.epoch_steps,
        validation_steps=args.valid_steps, # 100,
        callbacks=[
            tensorboard_callback,
            model_checkpoint_callback,
        ], 
        verbose=2
    )
    end_time = time.time()
    
    # val metrics
    val_keys = [v for v in layer_history.history.keys()]
    total_train_time = int((end_time - start_time) / 60)
    
    metrics_dict = {"total_train_time": total_train_time}
    logging.info(f"total_train_time: {total_train_time}")
    _ = [metrics_dict.update({key: layer_history.history[key][-1]}) for key in val_keys]
    
    # evaluate model
    if args.evaluate_model:
        logging.info(f"beginning model eval...")
        
        start_time = time.time()
        eval_dict = model.evaluate(valid_dataset, return_dict=True)
        end_time = time.time()

        total_eval_time = int((end_time - start_time) / 60)        
        logging.info(f"total_eval_time: {total_eval_time}")
        logging.info(f"eval_dict: {eval_dict}")
        
        # save eval dict
        if task_type == 'chief':
            logging.info(f"Chief saving model eval dict...")
            filehandler = open('model_eval_dict.pkl', 'wb')
            pkl.dump(eval_dict, filehandler)
            filehandler.close()
            train_utils.upload_blob(
                bucket_name=f'{OUTPUT_BUCKET}', 
                source_file_name='model_eval_dict.pkl', 
                destination_blob_name=f'{args.experiment_name}/{args.experiment_run}/combined-model-eval/model_eval_dict.pkl',
                project_id = project_number
            )
    
    # ====================================================
    # log Vertex Experiments
    # ====================================================
    SESSION_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=3)) # handle restarts for Vertex Experiments
    
    if task_type == 'chief':
        logging.info(f" task_type logging experiments: {task_type}")
        logging.info(f" task_id logging experiments: {task_id}")
        logging.info(f" logging data to experiment run: {args.experiment_run}-{SESSION_id}")
        
        with vertex_ai.start_run(
            f'{args.experiment_run}-{SESSION_id}', 
            # tensorboard=args.tb_resource_name
        ) as my_run:
            
            logging.info(f"logging metrics...")
            my_run.log_metrics(metrics_dict)

            logging.info(f"logging metaparams...")
            my_run.log_params(
                {
                    "layers": str(LAYER_SIZES),
                    "num_epochs": args.num_epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "valid_freq": args.valid_frequency,
                    "gpu_thread_cnt": args.tf_gpu_thread_count,
                    # "embed_freq": args.embed_frequency,
                    # "hist_freq": args.hist_frequency,
                }
            )

            vertex_ai.end_run()
            logging.info(f"EXPERIMENT RUN: {args.experiment_run}-{SESSION_id} has ended")

    # ====================================================
    # Save model
    # ====================================================
    MODEL_DIR_GCS_URI = f'{LOG_DIR}/model-dir'
    logging.info(f"Saving models to {MODEL_DIR_GCS_URI}")
    
    # save model from primary in multiworker
    if task_type == 'chief':
        
        # query tower
        query_model_dir = f"{MODEL_DIR_GCS_URI}/query_model"
        tf.saved_model.save(model.query_tower, export_dir=query_model_dir)
        logging.info(f'Saved chief query model to {query_model_dir}')
        
        # candidate tower
        candidate_model_dir = f"{MODEL_DIR_GCS_URI}/candidate_model"
        tf.saved_model.save(model.candidate_tower, export_dir=candidate_model_dir)
        logging.info(f'Saved chief candidate model to {candidate_model_dir}')
        
    else:
        # workers (tmp) 
        worker_dir_query = MODEL_DIR_GCS_URI + '/workertemp_query_/' + str(task_id)
        tf.io.gfile.makedirs(worker_dir_query)
        tf.saved_model.save(model.query_tower, worker_dir_query)
        logging.info(f'Saved worker: {task_id} query model to {worker_dir_query}')

        worker_dir_can = MODEL_DIR_GCS_URI + '/workertemp_can_/' + str(task_id)
        tf.io.gfile.makedirs(worker_dir_can)
        tf.saved_model.save(model.candidate_tower, worker_dir_can)
        logging.info(f'Saved worker: {task_id} candidate model to {worker_dir_can}')

    if task_type != 'chief':
        tf.io.gfile.rmtree(worker_dir_can)
        tf.io.gfile.rmtree(worker_dir_query)

    logging.info('Models saved') #all done

    # ====================================================
    # Save embeddings
    # ====================================================
    
    if args.write_embeddings:
        logging.info('Saving candidate embeddings...')
        local_candidate_embedding_index = 'candidate_embeddings.json'
    
        candidate_embeddings = parsed_candidate_dataset.batch(10000).map(
            lambda x: [
                x['track_uri_can'], 
                train_utils.tf_if_null_return_zero(
                    model.candidate_tower(x)
                )
            ]
        )
    
        # Save to the required format
        for batch in candidate_embeddings:
            songs, embeddings = batch
            with open(f"{local_candidate_embedding_index}", 'a') as f:
                for song, emb in zip(songs.numpy(), embeddings.numpy()):
                    f.write('{"id":"' + str(song) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}')
                    f.write("\n")

        if task_type == 'chief':
            train_utils.upload_blob(
                bucket_name=f'{OUTPUT_BUCKET}', 
                source_file_name=f'{local_candidate_embedding_index}', 
                destination_blob_name=f'{args.experiment_name}/{args.experiment_run}/candidates/{local_candidate_embedding_index}',
                project_id = project_number
            )
    
        logging.info(f"Saved {local_candidate_embedding_index} to {LOG_DIR}/candidates/{local_candidate_embedding_index}")


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
