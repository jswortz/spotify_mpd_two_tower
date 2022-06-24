# set variables
DROPOUT = False
DROPOUT_RATE = 0.2
EMBEDDING_DIM = 64
MAX_TOKENS = 100_000
BATCH_SIZE = 256
ARCH = [128, 64]
NUM_EPOCHS = 1
SEED = 41781897
PROJECT_ID = 'hybrid-vertex'
DROP_FIELDS = ['modified_at', 'row_number', 'seed_playlist_tracks']
N_RECORDS_PER_TFRECORD_FILE = 15000 #100ish mb  
TF_RECORDS_DIR = 'gs://spotify-tfrecords-blog'

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession
import warnings
warnings.filterwarnings("ignore") #do this b/c there's an info-level bug that can safely be ignored
import json
import tensorflow as tf
import tensorflow_recommenders as tfrs
import datetime
from tensorflow.python.lib.io import file_io
from tensorflow.train import BytesList, Feature, FeatureList, Int64List, FloatList
from tensorflow.train import SequenceExample, FeatureLists



def bq_to_tfdata(client, row_restriction, table_id, col_names, dataset, batch_size=BATCH_SIZE):
    TABLE_ID = table_id
    COL_NAMES = col_names
    DATASET = dataset
    bqsession = client.read_session(
        "projects/" + PROJECT_ID,
        PROJECT_ID, TABLE_ID, DATASET,
        COL_NAMES,
        requested_streams=2,
        row_restriction=row_restriction)
    dataset = bqsession.parallel_read_rows()
    return dataset.prefetch(1).shuffle(batch_size*10).batch(batch_size)

bq_2_tf_dict = {'name': {'mode': BigQueryClient.FieldMode.NULLABLE, 'output_type': dtypes.string},
 'collaborative': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'pid': {'mode': BigQueryClient.FieldMode.NULLABLE, 'output_type': dtypes.int64},
 'description': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'duration_ms_playlist': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.int64},
 'pid_pos_id': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'pos': {'mode':BigQueryClient.FieldMode.NULLABLE, 'output_type': dtypes.int64},
 'artist_name_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'track_uri_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'artist_uri_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'track_name_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'album_uri_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'duration_ms_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.float64},
 'album_name_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'track_pop_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.int64},
 'artist_pop_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.float64},
 'artist_genres_seed': {'mode': BigQueryClient.FieldMode.REPEATED,
  'output_type': dtypes.string},
 'artist_followers_seed': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.float64},
 'pos_seed_track': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.int64},
 'artist_name_seed_track': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'artist_uri_seed_track': {'mode':BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'track_name_seed_track': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'track_uri_seed_track': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'album_name_seed_track': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'album_uri_seed_track': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.string},
 'duration_seed_track': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.float64},
 'duration_ms_seed_pl': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.float64},
 'n_songs': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.int64},
 'num_artists': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.int64},
 'num_albums': {'mode': BigQueryClient.FieldMode.NULLABLE,
  'output_type': dtypes.int64},
'seed_playlist_tracks': {'mode': BigQueryClient.FieldMode.REPEATED,
  'output_type': dtypes.string}}

client = BigQueryClient()
batch_size = 1
bqsession = client.read_session(
        "projects/" + PROJECT_ID,
        PROJECT_ID, 'train', 'spotify_train_3',
        bq_2_tf_dict,
        requested_streams=2,)
dataset = bqsession.parallel_read_rows()
dataset = dataset.prefetch(1).shuffle(batch_size*10).batch(batch_size)

def main():
    for _ in dataset.take(1):
        return(print(_))
    
if __name__ == "__main__":
    main()

