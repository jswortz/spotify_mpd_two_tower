# import argparse
import gcsfs
import numpy as np

import tensorflow as tf

from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from pprint import pprint


# storage

BUCKET_NAME = 'spotify-beam-v3' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
VERSION = 'v3'
ROOT = f'gs://{BUCKET_NAME}/{VERSION}'

DATA_DIR = ROOT + '/data/' # Location to store data
STATS_DIR = ROOT +'/stats/' # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/' # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/' # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
CANDIDATE_DIR = ROOT + "/candidates/"


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
                    "track_name_can": _string_array(data['track_name_can']),
                    "artist_name_can": _string_array(data['artist_name_can']),
                    "album_name_can": _string_array(data['album_name_can']),
                    "track_uri_can": _string_array(data['track_uri_can']),
                    "duration_ms_can": _int64_feature(data['duration_ms_can']),     # TODO: likely to change to float
                    "track_pop_can": _int64_feature(data['track_pop_can']),         # TODO: likely to change to float
                    "artist_pop_can": _float_feature(data['artist_pop_can']),
                    "artist_genres_can": _string_array(data['artist_genres_can']),
                    "artist_followers_can": _float_feature(data['artist_followers_can']),
                }
            )
        )
        
        yield example

def run(args):
    '''
    define pipeline config
    '''
    
    BQ_TABLE = args['bq_source_table']
    CANDIDATE_SINK = args['candidate_sink']
    RUNNER = args['runner']
    NUM_TF_RECORDS = args['num_candidate_tfrecords']
    
    pipeline_options = beam.options.pipeline_options.GoogleCloudOptions(**args)
    # pipeline_options.view_as(SetupOptions).save_main_session = False #True
    print(pipeline_options)
    
    # Convert rows to tf-example
    _to_tf_example = candidates_to_tfexample(mode='candidates')
    
    # Write serialized example to tfrecords
    write_to_tf_record = beam.io.WriteToTFRecord(
        file_path_prefix = f'{CANDIDATE_DIR}/candidate-tracks', 
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


