
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
def test_model_endpoint(
    project: str,
    location: str,
    version: str,
    train_output_gcs_bucket: str,
    many_test_instances_gcs_filename: str,
    experiment_name: str,
    experiment_run: str,
    endpoint: str, # Input[Artifact],
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
    
    # LIST_OF_DICTS[200]
    
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
    
    
    # ====================================================
    # Send predictions
    # ====================================================
    # TOTAL_ROUNDS = 4
    SLEEP_SECONDS = 2 
    START=1
    END=4

    logging.info(f"testing online endpoint for {END} rounds")
    
    for i in range(START, END+1):
        
        count = 0

        for test in LIST_OF_DICTS:
            response = _endpoint.predict(instances=[test])

            if count > 0 and count % 250 == 0:
                logging.info(f"{count} prediciton requests..")

            count += 1
            
        logging.info(f"finsihed round {i} of {END}")
        time.sleep(SLEEP_SECONDS)
        
    logging.info(f"endpoint test complete - {count} predictions sent")
