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

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
MAX_WORKERS = '10'
RUNNER = 'DataflowRunner'
NETWORK = 'ucaip-haystack-vpc-network'


# CANDIDATE_DIR = ROOT + "/candidates/"

# estimate TF-Record shard count needed
# TF-Records


def _bytes_feature(value):
    """
    Get byte features
    """
    if type(value) == list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """
    Get int64 feature
    """
    if type(value) == list:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _string_array(value, shape=1):
    """
    Returns a bytes_list from a string / byte.
    """
    if type(value) == list:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8') for v in value]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))

def _float_feature(value, shape=1):
    """Returns a float_list from a float / double."""
    if type(value) == list:
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
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
        # Create Features
        # ===============================
        features = {
            # playlist - context features
            "name": _string_array(data['name']),
            'collaborative' : _string_array(data['collaborative']),
            'n_songs_pl' : _float_feature(data['n_songs_pl']),
            'num_artists_pl' : _float_feature(data['num_artists_pl']),
            'num_albums_pl' : _float_feature(data['num_albums_pl']),
            'description_pl' : _string_array(data['description_pl']),

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
            
            # Set List Types
            # Bytes
            "track_name_pl": _string_array(data['track_name_pl']),
            "artist_name_pl": _string_array(data['artist_name_pl']),
            "album_name_pl": _string_array(data['album_name_pl']),
            "track_uri_pl": _string_array(data['track_uri_pl']),
            "artist_genres_pl": _string_array(data['artist_genres_pl']),

            # Float List
            "duration_ms_songs_pl": _float_feature(data['duration_ms_songs_pl']),
            "artist_pop_pl": _float_feature(data['artist_pop_pl']),
            "artists_followers_pl": _float_feature(data['artists_followers_pl']),
            "track_pop_pl": _float_feature(data['track_pop_pl']),

        }
        

        yield tf.train.Example(features=tf.train.Features(feature=features))

def run(args):
    '''
    define pipeline config
    '''
    
    # storage
    VERSION = args['version']
    JOB_NAME = f'spotify-bq-tfrecords-{VERSION}-{TIMESTAMP}'

    ROOT = f'gs://{BUCKET_NAME}/{VERSION}'

    DATA_DIR = ROOT + '/data/' # Location to store data
    # STATS_DIR = ROOT +'/stats/' # Location to store stats 
    STAGING_DIR = ROOT + '/job/staging/' # Dataflow staging directory on GCP
    TEMP_DIR =  ROOT + '/job/temp/' # Dataflow temporary directory on GCP
    TF_RECORD_DIR = ROOT + '/tf-records/'
    
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
        num_shards=args['num_tfrecords']
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
