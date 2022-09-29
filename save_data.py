PROJECT_ID = 'hybrid-vertex'  # <--- TODO: CHANGE THIS
LOCATION = 'us-central1' 

TF_GPU_THREAD_MODE='gpu_private'

import json
import tensorflow as tf
import tensorflow_recommenders as tfrs

from google.cloud import storage

import numpy as np
import pickle as pkl
from pprint import pprint

from two_tower_src import two_tower as tt

batch_size = 512
train_dir = 'spotify-beam-v3'
train_dir_prefix = 'v3/train/'

valid_dir = 'spotify-beam-v3'
valid_dir_prefix = 'v3/valid/'

client = storage.Client()

train_files = []
for blob in client.list_blobs(f'{train_dir}', prefix=f'{train_dir_prefix}', delimiter="/"):
    train_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))

# OPTIMIZE DATA INPUT PIPELINE
train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
train_dataset = train_dataset.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
).map(
    tt.parse_tfrecord,
    num_parallel_calls=tf.data.AUTOTUNE,
).map(
    tt.return_padded_tensors,
    num_parallel_calls=tf.data.AUTOTUNE,)


valid_files = []
for blob in client.list_blobs(f'{valid_dir}', prefix=f'{valid_dir_prefix}', delimiter="/"):
    valid_files.append(blob.public_url.replace("https://storage.googleapis.com/", "gs://"))


# OPTIMIZE DATA INPUT PIPELINE
valid_dataset = tf.data.Dataset.from_tensor_slices(valid_files)
valid_dataset = valid_dataset.interleave(
    lambda x: tf.data.TFRecordDataset(x),
    cycle_length=tf.data.AUTOTUNE, 
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False
).map(
    tt.parse_tfrecord,
    num_parallel_calls=tf.data.AUTOTUNE,
).map(
    tt.return_padded_tensors,
    num_parallel_calls=tf.data.AUTOTUNE,)


from two_tower_src import preprocess as hash_data

train_hashed = train_dataset.map(hash_data.pre_hash_records, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

train_dir_hash = 'gs://spotify-beam-v3/v3/train-hashed/'

def custom_shard_func_train(element, n_shards=3000):
    x = element['track_uri_can'] % n_shards
    return(x)

checkpoint_prefix = "train_checkpoint"

step_counter = tf.Variable(0, trainable=False)
checkpoint_args = {
  "checkpoint_interval": 100_000,
  "step_counter": step_counter,
  "directory": checkpoint_prefix,
  "max_to_keep": 20,
}

tf.data.experimental.save(train_hashed, train_dir_hash, shard_func=custom_shard_func_train, checkpoint_args=checkpoint_args)