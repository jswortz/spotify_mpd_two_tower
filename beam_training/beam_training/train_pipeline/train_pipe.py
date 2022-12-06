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
BUCKET_NAME = 'spotify-data-regimes' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
# VERSION = 'v6'

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
MAX_WORKERS = '10'
RUNNER = 'DataflowRunner'
NETWORK = 'ucaip-haystack-vpc-network'

# Source data
BQ_TABLE = 'train_flatten_last_5_feats_v3'
BQ_DATASET = 'a_spotify_ds_1m'
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds


QUERY = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"



# CANDIDATE_DIR = ROOT + "/candidates/"

# estimate TF-Record shard count needed
# TF-Records

    
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
            'track_uri_pl',
            'track_name_pl',
            'artist_uri_pl',
            'artist_name_pl',
            'album_uri_pl',
            'album_name_pl',
            'duration_ms_songs_pl',
            'track_pop_pl',
            'artist_pop_pl',
            'artist_genres_pl',
            'artists_followers_pl',
            'tracks_playlist_titles_pl',
            'track_danceability_pl',
            'track_energy_pl',
            'track_key_pl',
            'track_loudness_pl',
            'track_mode_pl',
            'track_speechiness_pl',
            'track_acousticness_pl',
            'track_liveness_pl',
            'track_valence_pl',
            'track_tempo_pl',
            'time_signature_pl',
            
        ]

        ragged_dict = {}

        for _ in ragged_key_list:
            ragged_dict[_] = []
            
        for x in data['track_uri_pl']:
            ragged_dict['track_uri_pl'].append(x.encode('utf8'))

        for x in data['track_name_pl']:
            ragged_dict['track_name_pl'].append(x.encode('utf8'))
            
        for x in data['artist_uri_pl']:
            ragged_dict['artist_uri_pl'].append(x.encode('utf8'))

        for x in data['artist_name_pl']:
            ragged_dict['artist_name_pl'].append(x.encode('utf8'))

        for x in data['album_uri_pl']:
            ragged_dict['album_uri_pl'].append(x.encode('utf8'))
            
        for x in data['album_name_pl']:
            ragged_dict['album_name_pl'].append(x.encode('utf8'))

        for x in data['duration_ms_songs_pl']:
            ragged_dict['duration_ms_songs_pl'].append(x)
            
        for x in data['track_pop_pl']:
            ragged_dict['track_pop_pl'].append(x)

        for x in data['artist_pop_pl']:
            ragged_dict['artist_pop_pl'].append(x)
            
        for x in data['artist_genres_pl']:
            ragged_dict['artist_genres_pl'].append(x.encode('utf8'))

        for x in data['artists_followers_pl']:
            ragged_dict['artists_followers_pl'].append(x)

        for x in data['tracks_playlist_titles_pl']:
            ragged_dict['tracks_playlist_titles_pl'].append(x.encode('utf8'))
            
        for x in data['track_mode_pl']:
            ragged_dict['track_mode_pl'].append(x.encode('utf8'))
            
        for x in data['track_key_pl']:
            ragged_dict['track_key_pl'].append(x.encode('utf8'))
            
        for x in data['time_signature_pl']:
            ragged_dict['time_signature_pl'].append(x.encode('utf8'))
            
        for x in data['track_danceability_pl']:
            ragged_dict['track_danceability_pl'].append(x)

        for x in data['track_energy_pl']:
            ragged_dict['track_energy_pl'].append(x)
            
        for x in data['track_loudness_pl']:
            ragged_dict['track_loudness_pl'].append(x)
            
        for x in data['track_speechiness_pl']:
            ragged_dict['track_speechiness_pl'].append(x)
            
        for x in data['track_acousticness_pl']:
            ragged_dict['track_acousticness_pl'].append(x)
            
        for x in data['track_liveness_pl']:
            ragged_dict['track_liveness_pl'].append(x)
            
        for x in data['track_valence_pl']:
            ragged_dict['track_valence_pl'].append(x)
            
        for x in data['track_tempo_pl']:
            ragged_dict['track_tempo_pl'].append(x)

        # Set List Types
        # Bytes
        track_uri_pl = tf.train.BytesList(value=ragged_dict['track_uri_pl'])
        track_name_pl = tf.train.BytesList(value=ragged_dict['track_name_pl'])
        artist_uri_pl = tf.train.BytesList(value=ragged_dict['artist_uri_pl'])
        artist_name_pl = tf.train.BytesList(value=ragged_dict['artist_name_pl'])
        album_uri_pl = tf.train.BytesList(value=ragged_dict['album_uri_pl'])
        album_name_pl = tf.train.BytesList(value=ragged_dict['album_name_pl'])
        artist_genres_pl = tf.train.BytesList(value=ragged_dict['artist_genres_pl'])
        tracks_playlist_titles_pl = tf.train.BytesList(value=ragged_dict['tracks_playlist_titles_pl'])
        track_mode_pl = tf.train.BytesList(value=ragged_dict['track_mode_pl'])
        track_key_pl = tf.train.BytesList(value=ragged_dict['track_key_pl'])
        time_signature_pl = tf.train.BytesList(value=ragged_dict['time_signature_pl'])

        # Float List
        duration_ms_songs_pl = tf.train.FloatList(value=ragged_dict['duration_ms_songs_pl'])
        track_pop_pl = tf.train.FloatList(value=ragged_dict['track_pop_pl'])
        artist_pop_pl = tf.train.FloatList(value=ragged_dict['artist_pop_pl'])
        artists_followers_pl = tf.train.FloatList(value=ragged_dict['artists_followers_pl'])
        track_danceability_pl = tf.train.FloatList(value=ragged_dict['track_danceability_pl'])
        track_energy_pl = tf.train.FloatList(value=ragged_dict['track_energy_pl'])
        track_loudness_pl = tf.train.FloatList(value=ragged_dict['track_loudness_pl'])
        track_speechiness_pl = tf.train.FloatList(value=ragged_dict['track_speechiness_pl'])
        track_acousticness_pl = tf.train.FloatList(value=ragged_dict['track_acousticness_pl'])
        track_liveness_pl = tf.train.FloatList(value=ragged_dict['track_liveness_pl'])
        track_valence_pl = tf.train.FloatList(value=ragged_dict['track_valence_pl'])
        track_tempo_pl = tf.train.FloatList(value=ragged_dict['track_tempo_pl'])

        # Set FeatureLists
        # Bytes
        track_uri_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=track_uri_pl)])
        track_name_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=track_name_pl)])
        artist_uri_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=artist_uri_pl)])
        artist_name_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=artist_name_pl)])
        album_uri_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=album_uri_pl)])
        album_name_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=album_name_pl)])
        artist_genres_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=artist_genres_pl)])
        tracks_playlist_titles_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tracks_playlist_titles_pl)])
        track_mode_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=track_mode_pl)])
        track_key_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=track_key_pl)])
        time_signature_pl = tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=time_signature_pl)])

        # Float Lists
        duration_ms_songs_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=duration_ms_songs_pl)])
        track_pop_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_pop_pl)])
        artist_pop_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=artist_pop_pl)])
        artists_followers_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=artists_followers_pl)])
        track_danceability_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_danceability_pl)])
        track_energy_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_energy_pl)])
        track_loudness_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_loudness_pl)])
        track_speechiness_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_speechiness_pl)])
        track_acousticness_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_acousticness_pl)])
        track_liveness_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_liveness_pl)])
        track_valence_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_valence_pl)])
        track_tempo_pl = tf.train.FeatureList(feature=[tf.train.Feature(float_list=track_tempo_pl)])
        
        # ===============================
        # Create Context Features
        # ===============================
        context_features = {
            
            # ===================================================
            # # playlist - context features
            # ===================================================
            "pl_name_src": _string_array(data['pl_name_src']),
            'pl_collaborative_src' : _string_array(data['pl_collaborative_src']),
            'num_pl_followers_src' : _float_feature(data['num_pl_followers_src']),
            'pl_duration_ms_new' : _float_feature(data['pl_duration_ms_new']),
            'num_pl_songs_new' : _float_feature(data['num_pl_songs_new']),
            'num_pl_artists_new' : _float_feature(data['num_pl_artists_new']),
            'num_pl_albums_new' : _float_feature(data['num_pl_albums_new']),
            'avg_track_pop_pl_new' : _float_feature(data['avg_track_pop_pl_new']),
            'avg_artist_pop_pl_new' : _float_feature(data['avg_artist_pop_pl_new']),
            'avg_art_followers_pl_new' : _float_feature(data['avg_art_followers_pl_new']),
            'num_pl_artists_new' : _float_feature(data['num_pl_artists_new']),

            
            # ===================================================
            # candidate features
            # ===================================================
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
            "track_pl_titles_can": _string_array(data['track_pl_titles_can']),
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
            "track_time_signature_can": _string_array(data['track_time_signature_can']),
            
            # ===================================================
            # Set seed_tracks (list types)
            # ===================================================
            
#             # bytes / string
#             "track_uri_seed_track": _string_array(data['track_uri_seed_track']),
#             "track_name_seed_track": _string_array(data['track_name_seed_track']),
#             "artist_uri_seed_track": _string_array(data['artist_uri_seed_track']),
#             "artist_name_seed_track": _string_array(data['artist_name_seed_track']),
#             "album_uri_seed_track": _string_array(data['album_uri_seed_track']),
#             "album_name_seed_track": _string_array(data['album_name_seed_track']),
#             "artist_genres_seed_track": _string_array(data['artist_genres_seed_track']),
#             "track_pl_titles_seed_track": _string_array(data['track_pl_titles_seed_track']),
            
#             # Float List
#             "duration_seed_track": _float_feature(data['duration_seed_track']),
#             "track_pop_seed_track": _float_feature(data['track_pop_seed_track']),
#             "artist_pop_seed_track": _float_feature(data['artist_pop_seed_track']),
#             "artist_followers_seed_track": _float_feature(data['artist_followers_seed_track']),
#             "danceability_seed_track": _float_feature(data['danceability_seed_track']),
#             "key_seed_track": _float_feature(data['key_seed_track']),
#             "loudness_seed_track": _float_feature(data['loudness_seed_track']),
#             "mode_seed_track": _float_feature(data['mode_seed_track']),
#             "speechiness_seed_track": _float_feature(data['speechiness_seed_track']),
#             "acousticness_seed_track": _float_feature(data['acousticness_seed_track']),
#             "liveness_seed_track": _float_feature(data['liveness_seed_track']),
#             "valence_seed_track": _float_feature(data['valence_seed_track']),
#             "tempo_seed_track": _float_feature(data['tempo_seed_track']),
#             "time_signature_seed_track": _float_feature(data['time_signature_seed_track']),
            
            # ===================================================
            # Set playlist_seed_tracks (list types)
            # ===================================================
            
#             # bytes / string
#             "track_uri_pl": _string_array(data['track_uri_pl']),
#             "track_name_pl": _string_array(data['track_name_pl']),
#             "artist_uri_pl": _string_array(data['artist_uri_pl']),
#             "artist_name_pl": _string_array(data['artist_name_pl']),
#             "album_uri_pl": _string_array(data['album_uri_pl']),
#             "album_name_pl": _string_array(data['album_name_pl']),
#             "artist_genres_pl": _string_array(data['artist_genres_pl']),
#             "tracks_playlist_titles_pl": _string_array(data['tracks_playlist_titles_pl']),
            
#             # Float List
#             "duration_ms_songs_pl": _float_feature(data['duration_ms_songs_pl']),
#             "track_pop_pl": _float_feature(data['artist_pop_pl']),
#             "artist_pop_pl": _float_feature(data['artist_pop_pl']),
#             "artists_followers_pl": _float_feature(data['artists_followers_pl']),
#             "track_danceability_pl": _float_feature(data['track_danceability_pl']),
#             "track_key_pl": _float_feature(data['track_key_pl']),
#             "track_loudness_pl": _float_feature(data['track_loudness_pl']),
#             "track_mode_pl": _float_feature(data['track_mode_pl']),
#             "track_speechiness_pl": _float_feature(data['track_speechiness_pl']),
#             "track_acousticness_pl": _float_feature(data['track_acousticness_pl']),
#             "track_liveness_pl": _float_feature(data['track_liveness_pl']),
#             "track_valence_pl": _float_feature(data['track_valence_pl']),
#             "track_tempo_pl": _float_feature(data['track_tempo_pl']),
#             "time_signature_pl": _float_feature(data['time_signature_pl']),
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
                    "track_uri_pl": track_uri_pl,
                    "track_name_pl": track_name_pl,
                    "artist_uri_pl": artist_uri_pl,
                    'artist_name_pl': artist_name_pl,
                    "album_uri_pl": album_uri_pl,
                    'album_name_pl': album_name_pl,
                    "duration_ms_pl": duration_ms_pl,
                    "track_pop_pl": track_pop_pl,
                    "artist_pop_pl": artist_pop_pl,
                    "artist_genres_pl": artist_genres_pl,
                    "artists_followers_pl": artists_followers_pl,
                    'tracks_playlist_titles_pl': tracks_playlist_titles_pl,
                    'track_danceability_pl': track_danceability_pl,
                    'track_energy_pl': track_energy_pl,
                    'track_key_pl': track_key_pl,
                    'track_loudness_pl': track_loudness_pl,
                    'track_mode_pl': track_mode_pl,
                    'track_speechiness_pl': track_speechiness_pl,
                    'track_acousticness_pl': track_acousticness_pl,
                    'track_liveness_pl': track_liveness_pl,
                    'track_valence_pl': track_valence_pl,
                    'track_tempo_pl': track_tempo_pl,
                    'time_signature_pl': time_signature_pl,
                }
            )
        )

        yield seq

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
