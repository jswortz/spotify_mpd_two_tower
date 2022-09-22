# import argparse
import gcsfs
import numpy as np

import tensorflow as tf

from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


# setup
PROJECT_ID = 'hybrid-vertex'
BUCKET_NAME = 'spotify-beam-v3' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
VERSION = 'v6'

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
JOB_NAME = f'spotify-bq-tfrecords-{VERSION}-{TIMESTAMP}'
MAX_WORKERS = '10'
RUNNER = 'DataflowRunner'
NETWORK = 'ucaip-haystack-vpc-network'

# Source data
BQ_TABLE = 'train_flatten'
BQ_DATASET = 'mdp_eda_test'
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds


QUERY = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"

# storage
# setup
PROJECT_ID = 'hybrid-vertex'
BUCKET_NAME = 'spotify-beam-v3' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
VERSION = 'v3'

# storage
ROOT = f'gs://{BUCKET_NAME}/{VERSION}'

DATA_DIR = ROOT + '/data/' # Location to store data
# STATS_DIR = ROOT +'/stats/' # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/' # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/' # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
# CANDIDATE_DIR = ROOT + "/candidates/"

# estimate TF-Record shard count needed
# TF-Records
total_samples = 65_346_428  
samples_per_file = 12_800 
NUM_TF_RECORDS = total_samples // samples_per_file

if NUM_TF_RECORDS % total_samples:
    NUM_TF_RECORDS += 1
    
print("Number of Expected TFRecords: {}".format(NUM_TF_RECORDS)) # 5343


def _bytes_feature(value):
    """
    Get byte features
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """
    Get int64 feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _string_array(value):
    """
    Returns a bytes_list from a string / byte.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class TrainTfSeqExampleDoFn(beam.DoFn):
    """
    Convert training sample into TFExample
    """
    def __init__(self, task):
        """
        Initialization
        """
        self.task = task

    
    def process(self, data):
        """
        Convert BQ row to tf-example
        """
    
        # ===============================
        # Ragged Features - Query
        # ===============================
        ragged_key_list = [
            'track_name_pl',
            'artist_name_pl',
            'album_name_pl',
            'track_uri_pl',
            'duration_ms_songs_pl',
            'artist_pop_pl',
            'artists_followers_pl',
            'track_pop_pl',
            'artist_genres_pl',
        ]

        ragged_dict = {}

        for _ in ragged_key_list:
            ragged_dict[_] = []

        for x in data['track_name_pl']:
            ragged_dict['track_name_pl'].append(x.encode('utf8'))

        for x in data['artist_name_pl']:
            ragged_dict['artist_name_pl'].append(x.encode('utf8'))

        for x in data['album_name_pl']:
            ragged_dict['album_name_pl'].append(x.encode('utf8'))

        for x in data['track_uri_pl']:
            ragged_dict['track_uri_pl'].append(x.encode('utf8'))

        for x in data['duration_ms_songs_pl']:
            ragged_dict['duration_ms_songs_pl'].append(x)

        for x in data['artist_pop_pl']:
            ragged_dict['artist_pop_pl'].append(x)

        for x in data['artists_followers_pl']:
            ragged_dict['artists_followers_pl'].append(x)

        for x in data['track_pop_pl']:
            ragged_dict['track_pop_pl'].append(x)

        for x in data['artist_genres_pl']:
            ragged_dict['artist_genres_pl'].append(x.encode('utf8'))

        # Set List Types
        # Bytes
        track_name_pl = tf.train.BytesList(value=ragged_dict['track_name_pl'])
        artist_name_pl = tf.train.BytesList(value=ragged_dict['artist_name_pl'])
        album_name_pl = tf.train.BytesList(value=ragged_dict['album_name_pl'])
        track_uri_pl = tf.train.BytesList(value=ragged_dict['track_uri_pl'])
        artist_genres_pl = tf.train.BytesList(value=ragged_dict['artist_genres_pl'])

        # Float List
        duration_ms_songs_pl = tf.train.FloatList(value=ragged_dict['duration_ms_songs_pl'])
        artist_pop_pl = tf.train.FloatList(value=ragged_dict['artist_pop_pl'])
        artists_followers_pl = tf.train.FloatList(value=ragged_dict['artists_followers_pl'])
        track_pop_pl = tf.train.FloatList(value=ragged_dict['track_pop_pl'])

        # Set FeatureLists
        # Bytes
        track_name_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=track_name_pl)])
        artist_name_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=artist_name_pl)])
        album_name_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=album_name_pl)])
        track_uri_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=track_uri_pl)])
        artist_genres_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=artist_genres_pl)])

        # Float Lists
        duration_ms_songs_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=duration_ms_songs_pl)])
        artist_pop_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=artist_pop_pl)])
        artists_followers_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=artists_followers_pl)])
        track_pop_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_pop_pl)])
        
        # ===============================
        # Create Context Features
        # ===============================
        context_features = {
            # playlist - context features
            "name": _string_array(data['name']),
            'collaborative' : _string_array(data['collaborative']),
            # 'duration_ms_seed_pl' : _float_feature(data['duration_ms_seed_pl']),
            'n_songs_pl' : _float_feature(data['n_songs_pl']),
            'num_artists_pl' : _float_feature(data['num_artists_pl']),
            'num_albums_pl' : _float_feature(data['num_albums_pl']),
            'description_pl' : _string_array(data['description_pl']),

            # seed track - context features
            'track_name_seed_track' : _string_array(data['track_name_seed_track']),
            'artist_name_seed_track' : _string_array(data['artist_name_seed_track']),
            'album_name_seed_track' : _string_array(data['album_name_seed_track']),
            'track_uri_seed_track' : _string_array(data['track_uri_seed_track']),
            'artist_uri_seed_track' : _string_array(data['artist_uri_seed_track']),
            'album_uri_seed_track' : _string_array(data['album_uri_seed_track']),
            'duration_seed_track' : _float_feature(data['duration_seed_track']),
            'track_pop_seed_track' : _float_feature(data['track_pop_seed_track']),
            'artist_pop_seed_track' : _float_feature(data['artist_pop_seed_track']),
            'artist_genres_seed_track' : _string_array(data['artist_genres_seed_track']),
            'artist_followers_seed_track' : _float_feature(data['artist_followers_seed_track']),

            #candidate features
            "track_name_can": _string_array(data['track_name_can']), 
            "artist_name_can": _string_array(data['artist_name_can']),
            "album_name_can": _string_array(data['album_name_can']),
            "track_uri_can": _string_array(data['track_uri_can']),
            "artist_uri_can": _string_array(data['artist_uri_can']),
            "album_uri_can": _string_array(data['album_uri_can']),
            "duration_ms_can": _float_feature(data['duration_ms_can']),
            "track_pop_can": _float_feature(data['track_pop_can']), 
            "artist_pop_can": _float_feature(data['artist_pop_can']),
            "artist_genres_can": _string_array(data['artist_genres_can']),
            "artist_followers_can": _float_feature(data['artist_followers_can']),
        }
        
        # ===============================
        # Create Sequence
        # ===============================
        seq = tf.train.SequenceExample(
            context=tf.train.Features(
                feature=context_features
            ),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    "track_name_pl": track_name_pl,
                    "artist_name_pl": artist_name_pl,
                    "album_name_pl": album_name_pl,
                    "track_uri_pl": track_uri_pl,
                    "duration_ms_songs_pl": duration_ms_songs_pl,
                    "artist_pop_pl": artist_pop_pl,
                    "artists_followers_pl": artists_followers_pl,
                    "track_pop_pl": track_pop_pl,
                    "artist_genres_pl": artist_genres_pl
                }
            )
        )

        yield seq

def run(args):
    '''
    define pipeline config
    '''
    
    pipeline_args = [
        '--runner', RUNNER,
        '--network', NETWORK,
        '--region', REGION,
        '--project', PROJECT_ID,
        '--staging_location', STAGING_DIR,
        '--temp_location', TEMP_DIR,
        # '--template_location', TEMPLATE_LOCATION,
        '--job_name', JOB_NAME,
        '--num_workers', MAX_WORKERS,
        '--setup_file', './setup.py',
        # '--requirements_file', 'requirements.txt',
        # '--worker_machine_type','xxx'
    ]
    
    pipeline_options = beam.options.pipeline_options.GoogleCloudOptions(pipeline_args)
    # pipeline_options.view_as(SetupOptions).save_main_session = False #True
    print(pipeline_options)
    
    # Convert rows to tf-example
    _to_tf_example = TrainTfSeqExampleDoFn(task="train")
    
    # Write serialized example to tfrecords
    write_to_tf_record = beam.io.WriteToTFRecord(
        file_path_prefix = f'{ROOT}/{args["folder"]}/', 
        file_name_suffix=".tfrecords",
        num_shards=NUM_TF_RECORDS
    )

    with beam.Pipeline(RUNNER, options=pipeline_options) as pipeline:
        (pipeline 
         # | "Read from BigQuery">> beam.io.Read(beam.io.BigQuerySource(table=TABLE_SPEC, flatten_results=True))
         | 'Read from BigQuery' >> beam.io.ReadFromBigQuery(table=args['bq_source_table'])
         | 'Convert to tf Example' >> beam.ParDo(_to_tf_example)
         | 'Serialize to String' >> beam.Map(lambda example: example.SerializeToString(deterministic=True))
         | "Write as TFRecords to GCS" >> write_to_tf_record
        )

if __name__ == "__main__":
    main()
