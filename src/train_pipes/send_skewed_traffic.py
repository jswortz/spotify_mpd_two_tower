
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (
    Artifact, Dataset, Input, InputPath, 
    Model, Output, OutputPath, component, Metrics
)
@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.26.1',
        'google-cloud-pipeline-components',
        'google-cloud-storage',
        'tensorflow==2.11.0',
        'numpy'
    ],
)
def send_skewed_traffic(
    project: str,
    location: str,
    version: str,
    train_output_gcs_bucket: str,
    experiment_name: str,
    experiment_run: str,
    endpoint: str, # Input[Artifact],
    many_test_instances_gcs_filename: str,
    # feature_dict: dict,
    # metrics: Output[Metrics],
):
    
    import logging
    from datetime import datetime
    import time
    import numpy as np
    import pickle as pkl
    
    from google.cloud import aiplatform as vertex_ai
    
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob
    
    from google.protobuf import json_format
    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Value
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    import tensorflow as tf

    logging.getLogger().setLevel(logging.INFO)

    vertex_ai.init(
        project=project,
        location=location,
    )
    storage_client = storage.Client(project=project)
    
    # ====================================================
    # helper functions
    # ====================================================
    def download_blob(bucket_name, source_gcs_obj, local_filename):
        """Uploads a file to the bucket."""
        # storage_client = storage.Client(project=project_number)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_gcs_obj)
        blob.download_to_filename(local_filename)
        
        filehandler = open(f'{local_filename}', 'rb')
        loaded_dict = pkl.load(filehandler)
        filehandler.close()
        
        logging.info(f"File {local_filename} downloaded from gs://{bucket_name}/{source_gcs_obj}")
        
        return loaded_dict
    
    # ====================================================
    # get deployed model endpoint
    # ====================================================
    logging.info(f"Endpoint = {endpoint}")
    gcp_resources = Parse(endpoint, GcpResources())
    logging.info(f"gcp_resources = {gcp_resources}")
    
    _endpoint_resource = gcp_resources.resources[0].resource_uri
    logging.info(f"_endpoint_resource = {_endpoint_resource}")
    
    _endpoint_uri = "/".join(_endpoint_resource.split("/")[-8:-2])
    logging.info(f"_endpoint_uri = {_endpoint_uri}")
    
    # define endpoint resource in component
    _endpoint = vertex_ai.Endpoint(_endpoint_uri)
    logging.info(f"_endpoint defined")
    
    # ===================================================
    # load test instance
    # ===================================================
    LOCAL_INSTANCE_FILE = 'test_instance_list.pkl'
    GCS_PATH_TO_BLOB = f'{experiment_name}/{experiment_run}/{many_test_instances_gcs_filename}'
    LOADED_TEST_LIST = f'loaded_{LOCAL_INSTANCE_FILE}'
    
    loaded_test_instance = download_blob(
        bucket_name=train_output_gcs_bucket,
        source_gcs_obj=GCS_PATH_TO_BLOB,
        local_filename=LOADED_TEST_LIST
    )
    logging.info(f'loaded_test_instance: {loaded_test_instance}')
    
    filehandler = open(LOADED_TEST_LIST, 'rb')
    LIST_OF_DICTS = pkl.load(filehandler)
    filehandler.close()
    
    logging.info(f'len(LIST_OF_DICTS): {len(LIST_OF_DICTS)}')
    
    # ====================================================
    # load skew features stats
    # ====================================================
    SKEW_FEATURES_STATS_FILE = 'skew_feat_stats.pkl'
    GCS_PATH_TO_BLOB = f'{experiment_name}/{experiment_run}/{SKEW_FEATURES_STATS_FILE}'
    LOADED_SKEW_FEATURES_STATS_FILE = f"loaded_{SKEW_FEATURES_STATS_FILE}"
    logging.info(f'loading: {LOADED_SKEW_FEATURES_STATS_FILE}')
    
    loaded_skew_test_instance = download_blob(
        bucket_name=train_output_gcs_bucket,
        source_gcs_obj=GCS_PATH_TO_BLOB,
        local_filename=LOADED_SKEW_FEATURES_STATS_FILE
    )
    logging.info(f'loaded_skew_test_instance: {loaded_skew_test_instance}')
    
    filehandler_v2 = open(LOADED_SKEW_FEATURES_STATS_FILE, 'rb')
    SKEW_FEATURES = pkl.load(filehandler_v2)
    filehandler_v2.close()
    
    mean_durations, std_durations = SKEW_FEATURES['pl_duration_ms_new']
    mean_num_songs, std_num_songs = SKEW_FEATURES['num_pl_songs_new']
    mean_num_artists, std_num_artists = SKEW_FEATURES['num_pl_artists_new']
    mean_num_albums, std_num_albums = SKEW_FEATURES['num_pl_albums_new']
    
    logging.info(f"std_durations   : {round(std_durations, 0)}")
    logging.info(f"std_num_songs   : {round(std_num_songs, 0)}")
    logging.info(f"std_num_artists : {round(std_num_artists, 0)}")
    logging.info(f"std_num_albums  : {round(std_num_albums, 0)}\n")
    
    def monitoring_test(endpoint, instances, skew_feat_stat, start=2, end=4):

        mean_durations, std_durations = skew_feat_stat['pl_duration_ms_new']
        mean_num_songs, std_num_songs = skew_feat_stat['num_pl_songs_new']
        mean_num_artists, std_num_artists = skew_feat_stat['num_pl_artists_new']
        mean_num_albums, std_num_albums = skew_feat_stat['num_pl_albums_new']
        
        logging.info(f"std_durations   : {round(std_durations, 0)}")
        logging.info(f"std_num_songs   : {round(std_num_songs, 0)}")
        logging.info(f"std_num_artists : {round(std_num_artists, 0)}")
        logging.info(f"std_num_albums  : {round(std_num_albums, 0)}\n")

        total_preds = 0

        for multiplier in range(start, end+1):

            print(f"multiplier: {multiplier}")

            pred_count = 0

            for example in instances:
                list_dict = {}

                example['pl_duration_ms_new'] = round(std_durations * multiplier, 0)
                example['num_pl_songs_new'] = round(std_num_songs * multiplier, 0)
                example['num_pl_artists_new'] = round(std_num_artists * multiplier, 0)
                example['num_pl_albums_new'] = round(std_num_albums * multiplier, 0)
                # list_of_skewed_instances.append(example)

                response = endpoint.predict(instances=[example])

                if pred_count > 0 and pred_count % 250 == 0:
                    print(f"pred_count: {pred_count}")

                pred_count += 1
                total_preds += 1

            logging.info(f"sent {pred_count} pred requests with {multiplier}X multiplier")

        logging.info(f"sent {total_preds} total pred requests")
        
    # send skewed traffic
    monitoring_test(
        endpoint=_endpoint, 
        instances=LIST_OF_DICTS,
        skew_feat_stat=SKEW_FEATURES,
        start=2, 
        end=8
    )
