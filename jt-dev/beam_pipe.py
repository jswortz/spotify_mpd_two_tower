# import argparse
import gcsfs
import numpy as np

import tensorflow as tf

from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from pprint import pprint

# setup
PROJECT_ID = 'hybrid-vertex'
BUCKET_NAME = 'spotify-beam-v1' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
VERSION = 'v3'

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
JOB_NAME = f'spotify-bq-tfrecords-{VERSION}-{TIMESTAMP}'
MAX_WORKERS = '20'
RUNNER = 'DataflowRunner'
NETWORK = 'ucaip-haystack-vpc-network'

# Source data
BQ_TABLE = 'candidate_features_v3'
BQ_DATASET = 'spotify_train_4'
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds

# storage
ROOT = f'gs://{BUCKET_NAME}/{VERSION}'

DATA_DIR = ROOT + '/data/' # Location to store data
STATS_DIR = ROOT +'/stats/' # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/' # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/' # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
CANDIDATE_DIR = ROOT + "/candidates/"

QUERY = f"SELECT * FROM {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"

NUM_TF_RECORDS = 8



class candidates_to_tfexample(beam.DoFn):
    '''
    convert bigqury rows to tf.examples
    '''
    def __init__(self, mode):
        """
          Initialization
        """
        self.mode = mode

    @staticmethod
    def _int64_feature(value):
        """
        Get int64 feature
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    @staticmethod
    def _string_array(value):
        """
        Returns a bytes_list from a string / byte.
        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode('utf-8') for v in value]))
    
    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in value]))
    
    def process(self, data):
        """
        Convert BQ row to tf-example
        """
        
        example = tf.train.Example(
            features=tf.train.Features(
                features = {
                    "track_name": _string_array(data['track_name']),
                    "artist_name": _string_array(data['artist_name']),
                    "album_name": _string_array(data['album_name']),
                    "track_uri": _string_array(data['track_uri']),
                    "duration_ms": _int64_feature(data['duration_ms']),     # TODO: likely to change to float
                    "track_pop": _int64_feature(data['track_pop']),         # TODO: likely to change to float
                    "artist_pop": _float_feature(data['artist_pop']),
                    "artist_genres": _string_array(data['artist_genres']),
                    "artist_followers": _float_feature(data['artist_followers']),
                }
            )
        )
        
        yield example
        
def main():
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
        '--setup_file', './setup.py'
    ]
    
    pipeline_options = beam.options.pipeline_options.GoogleCloudOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    print(pipeline_options)
    
    # Convert rows to tf-example
    _to_tf_example = candidates_to_tfexample(mode='candidates')
    
    # Write serialized example to tfrecords
    write_to_tf_record = beam.io.WriteToTFRecord(
        file_path_prefix = f'{CANDIDATE_DIR}/candidate-tracks-', 
        file_name_suffix=".tfrecords",
        num_shards=NUM_TF_RECORDS
    )

    with beam.Pipeline(RUNNER, options=pipeline_options) as pipeline:
        (pipeline 
         | "Read from BigQuery">> beam.io.Read(beam.io.BigQuerySource(table=TABLE_SPEC, flatten_results=True))
         | 'Convert to tf Example' >> beam.ParDo(_to_tf_example)
         | 'Serialize to String' >> beam.Map(lambda example: example.SerializeToString(deterministic=True))
         | "Write as TFRecords to GCS" >> write_to_tf_record
        )
        
if __name__ == "__main__":
    main()