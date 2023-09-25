import os
# import argparse
import gcsfs
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

from datetime import datetime

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def _int64_feature(value):
    """
    Get int64 feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def _string_array(value):
    """
    Returns a bytes_list from a string / byte.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value).encode('utf-8')]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    

class candidates_to_tfexample(beam.DoFn):
    '''
    convert bigqury rows to tf.examples
    '''
    def __init__(self, mode):
        """
          Initialization
        """
        self.mode = mode
    
    
    def process(self, data):
        """
        Convert BQ row to tf-example
        """
        
        example = tf.train.Example(
            features=tf.train.Features(
                feature = {
                    "track_uri_can": _string_array(data['track_uri_can']),
                    "track_name_can": _string_array(data['track_name_can']),
                    "artist_uri_can": _string_array(data['artist_uri_can']),
                    "artist_name_can": _string_array(data['artist_name_can']),
                    "album_uri_can": _string_array(data['album_uri_can']),
                    "album_name_can": _string_array(data['album_name_can']),
                    "duration_ms_can": _float_feature(data['duration_ms_can']),   
                    "track_pop_can": _float_feature(data['track_pop_can']),         
                    "artist_pop_can": _float_feature(data['artist_pop_can']),
                    "artist_genres_can": _string_array(data['artist_genres_can']),
                    "artist_followers_can": _float_feature(data['artist_followers_can']),
                    # new
                    # "track_pl_titles_can": _string_array(data['track_pl_titles_can']),
                    "track_danceability_can": _float_feature(data['track_danceability_can']),
                    "track_energy_can": _float_feature(data['track_energy_can']),
                    "track_key_can": _string_array(data['track_key_can']),
                    "track_loudness_can": _float_feature(data['track_loudness_can']),
                    "track_mode_can": _string_array(data['track_mode_can']),
                    "track_speechiness_can": _float_feature(data['track_speechiness_can']),
                    "track_acousticness_can": _float_feature(data['track_acousticness_can']),
                    "track_instrumentalness_can": _float_feature(data['track_instrumentalness_can']),
                    "track_liveness_can": _float_feature(data['track_liveness_can']),
                    "track_valence_can": _float_feature(data['track_valence_can']),
                    "track_tempo_can": _float_feature(data['track_tempo_can']),
                    "track_time_signature_can": _string_array(data['time_signature_can']),

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
    QUERY = args['source_query']
    
    pipeline_options = beam.options.pipeline_options.GoogleCloudOptions(**args)
    print(pipeline_options)
    
    # Convert rows to tf-example
    _to_tf_example = candidates_to_tfexample(mode='candidates')
    
    # Write serialized example to tfrecords
    write_to_tf_record = beam.io.WriteToTFRecord(
        file_path_prefix = CANDIDATE_SINK, 
        file_name_suffix=".tfrecords",
        num_shards=1 #hardcoding due to smaller size
    )

    with beam.Pipeline(RUNNER, options=pipeline_options) as pipeline:
        (pipeline 
         | 'Read from BigQuery' >> beam.io.ReadFromBigQuery(table=BQ_TABLE, flatten_results=True)
         | 'Convert to tf Example' >> beam.ParDo(_to_tf_example)
         | 'Serialize to String' >> beam.Map(lambda example: example.SerializeToString(deterministic=True))
         | "Write as TFRecords to GCS" >> write_to_tf_record
        )


