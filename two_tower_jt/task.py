import os

os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
from absl import app
from absl import flags
from absl import logging


import json

import tensorflow as tf
import logging
import time

import tensorflow_recommenders as tfrs

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


from google.cloud import storage

import two_tower as tt #import the model from the same module

##########################
# #########################
# ##### args ##############
# #########################
# #########################
# #########################

FLAGS = flags.FLAGS
flags.DEFINE_string("train_dir", 'spotify-beam-v3', "bucket where tfrecords live")
flags.DEFINE_string("train_dir_prefix", 'v1/train_last_5/','path to training data in train_dir')
flags.DEFINE_string("valid_dir_prefix", 'v1/valid_last_5/','path to validation data in train_dir')
flags.DEFINE_string("EXPERIMENT_NAME", None,'vertex ai experiment name')


flags.DEFINE_string("OUTPUT_PATH", 'gs://two-tower-models','location saved models and embeddings')
flags.DEFINE_integer("SEED", 41781897, "random seed")


flags.DEFINE_float("LR", 0.1, "Learning Rate")
flags.DEFINE_bool("DROPOUT", False, "Use Dropout - T/F bool type")
flags.DEFINE_float("DROPOUT_RATE", 0.4, "Dropout rate only works with DROPOUT=True")
flags.DEFINE_integer("EMBEDDING_DIM", 128, "Embedding dimension")
flags.DEFINE_string("ARCH", None, "deep architecture, expressed as a list of ints in string format - will be parsed into list")

flags.DEFINE_integer("NUM_EPOCHS", None, "Number of epochs")
flags.DEFINE_integer("BATCH_SIZE", None, "batch size")

flags.DEFINE_integer("MAX_TOKENS", 50000, "Max embeddings for query and last_n tracks")
flags.DEFINE_string("PROJECT_ID", "hybrid-vertex", "Project ID")
flags.DEFINE_string("LOCATION", "us-central1", "GCP Location")


##########################
# #########################
# ##### helper functions ##
# #########################
# #########################
# #########################

# create a tf function to convert any bad null values
TB_RESOURCE_NAME = 'projects/934903580331/locations/us-central1/tensorboards/7336372589079560192' #fqn - project number then tensorboard id

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




##########################
# #########################
# #######data loading######
# #########################
# #########################
# #########################

def main(argv):
    
    
    invoke_time = time.strftime("%Y%m%d-%H%M%S")
    EXPERIMENT_NAME = FLAGS.EXPERIMENT_NAME
    RUN_NAME = EXPERIMENT_NAME+'run'+time.strftime("%Y%m%d-%H%M%S")
    

    path = FLAGS.OUTPUT_PATH
    LOG_DIR = path+"/tb-logs/"+EXPERIMENT_NAME

    batch_size = FLAGS.BATCH_SIZE
    train_dir = FLAGS.train_dir
    train_dir_prefix = FLAGS.train_dir_prefix

    valid_dir = FLAGS.train_dir
    valid_dir_prefix = FLAGS.valid_dir_prefix

    client = storage.Client()
    from google.cloud import aiplatform as vertex_ai
    vertex_ai.init(project=FLAGS.PROJECT_ID,
                   location=FLAGS.LOCATION,
                   experiment=EXPERIMENT_NAME)


    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO


    train_files = []
    for blob in client.list_blobs(f'{train_dir}', prefix=f'{train_dir_prefix}', delimiter="/"):
        train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

    def full_parse(data):
        # used for interleave - takes tensors and returns a tf.dataset
        data = tf.data.TFRecordDataset(data)
        return data

    train_dataset = tf.data.Dataset.from_tensor_slices(train_files).prefetch(
        tf.data.AUTOTUNE,
    )

    train_dataset = train_dataset.interleave(
        full_parse,
        cycle_length=tf.data.AUTOTUNE, 
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    ).shuffle(batch_size*4, reshuffle_each_iteration=False).map(tt.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE,).batch(
        batch_size 
    ).prefetch(
        tf.data.AUTOTUNE,
    ).with_options(options)


    valid_files = []
    for blob in client.list_blobs(f'{valid_dir}', prefix=f'{valid_dir_prefix}', delimiter="/"):
        valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))


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

    ##########################
    ##########################
    #######model creation#####
    ##########################
    ##########################
    ##########################

    layer_sizes=get_arch_from_string(FLAGS.ARCH)

    model = tt.TheTwoTowers(layer_sizes)

    LR = FLAGS.LR
    opt = tf.keras.optimizers.Adagrad(LR)
    model.compile(optimizer=opt)
    
    def get_upload_logs_to_manged_tb_command(ttl_hrs, oneshot="false"):
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
    
    # we are going to ecapsulate this one-shot log uploader via a custom callback:

    class UploadTBLogsBatchEnd(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            os.system(get_upload_logs_to_manged_tb_command(ttl_hrs = 5, oneshot="true"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=0, 
            write_graph=True, 
            # profile_batch=(20,50) #run profiler on steps 20-40 - enable this line if you want to run profiler from the utils/ notebook
        )


    ##########################
    ##########################
    #######training###########
    ##########################
    ##########################
    ##########################

    NUM_EPOCHS = FLAGS.NUM_EPOCHS
    RUN_NAME = f'run-{EXPERIMENT_NAME}-{time.strftime("%Y%m%d-%H%M%S")}'#be sure to think about run and experiment naming strategies so names don't collide

    #start the run to collect metrics - note `.log_parameters()` is available but not used

    #start the timer and training
    start_time = time.time()
    layer_history = model.fit(
        train_dataset.unbatch().batch(batch_size),
        validation_data=valid_dataset,
        validation_freq=3,
        epochs=NUM_EPOCHS,
        # steps_per_epoch=2, #use this for development to run just a few steps
        validation_steps = 100,
        callbacks=[tensorboard_callback,
                   UploadTBLogsBatchEnd()], #the tensorboard will be automatically associated with the experiment and log subsequent runs with this callback
        verbose=1
    )

    end_time = time.time()
    val_keys = [v for v in layer_history.history.keys()]
    runtime_mins = int((end_time - start_time) / 60)

    
    vertex_ai.start_run(RUN_NAME, tensorboard=TB_RESOURCE_NAME)

    vertex_ai.log_params({"layers": str(layer_sizes), 
                            "num_epochs": NUM_EPOCHS,
                            "batch_size": batch_size,
                         })

    #gather the metrics for the last epoch to be saved in metrics
    metrics_dict = {"train-time-minutes": runtime_mins}
    _ = [metrics_dict.update({key: layer_history.history[key][-1]}) for key in val_keys]
    vertex_ai.log_metrics(metrics_dict)
    vertex_ai.end_run()


    ##########################
    ##########################
    #######save the models####
    ##########################
    ##########################
    ##########################


    tf.saved_model.save(model.query_tower, export_dir=FLAGS.OUTPUT_PATH + "/query_model")
    tf.saved_model.save(model.candidate_tower, export_dir=FLAGS.OUTPUT_PATH + "/candidate_model")

    ##########################
    ##########################
    #######save embeddings####
    ##########################
    ##########################
    ##########################

    candidate_embeddings = tt.parsed_candidate_dataset.batch(10000).map(lambda x: [x['track_uri_can'], tf_if_null_return_zero(model.candidate_tower(x))])

    # Save to the required format
    # make sure you start out with a clean empty file for the append write

    for batch in candidate_embeddings:
        songs, embeddings = batch
        with open(f"candidate_embeddings_{invoke_time}.json", 'a') as f:
            for song, emb in zip(songs.numpy(), embeddings.numpy()):
                f.write('{"id":"' + str(song) + '","embedding":[' + ",".join(str(x) for x in list(emb)) + ']}')
                f.write("\n")

    tt.upload_blob('two-tower-models', f'candidate_embeddings_{invoke_time}.json', f'candidates/candidate_embeddings_{invoke_time}.json')


if __name__ == "__main__":
    app.run(main)
