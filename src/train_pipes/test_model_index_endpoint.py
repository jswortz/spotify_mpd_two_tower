
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
def test_model_index_endpoint(
    project: str,
    location: str,
    version: str,
    train_output_gcs_bucket: str,
    test_instances_gcs_filename: str,
    experiment_name: str,
    experiment_run: str,
    # train_dir: str,
    # train_dir_prefix: str,
    # ann_index_resource_uri: str,
    ann_index_endpoint_resource_uri: str,
    brute_index_endpoint_resource_uri: str,
    gcs_train_script_path: str,
    endpoint: str, # Input[Artifact],
    metrics: Output[Metrics],
):
    
    import logging
    from datetime import datetime
    import time
    import numpy as np
    import pickle as pkl
    
    import base64

    from typing import Dict, List, Union

    from google.cloud import aiplatform as vertex_ai
    
    from google.protobuf import json_format
    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Value

    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources
    
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob

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
    
    # ==============================================================
    # helper function for returning endpoint predictions via json
    # ==============================================================
    
    def predict_custom_trained_model_sample(
        project: str,
        endpoint_id: str,
        instances: Dict,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
    ):
        """
        either single instance of type dict or a list of instances.
        This client only needs to be created once, and can be reused for multiple requests.
        """

        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": api_endpoint}
        
        # Initialize client that will be used to create and send requests.
        client = vertex_ai.gapic.PredictionServiceClient(client_options=client_options)
        
        # The format of each instance should conform to the deployed model's prediction input schema.
        instances = instances if type(instances) == list else [instances]
        instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        
        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )
        
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        logging.info(f'Response: {response}')
        logging.info(f'Deployed Model ID(s): {response.deployed_model_id}')
        # The predictions are a google.protobuf.Value representation of the model's predictions.
        _predictions = response.predictions
        logging.info(f'Response Predictions: {_predictions}')
        
        return _predictions
    
    # ===================================================
    # load test instance
    # ===================================================
    LOCAL_TEST_INSTANCE = 'test_instances_dict.pkl'
    GCS_PATH_TO_BLOB = f'{experiment_name}/{experiment_run}/{test_instances_gcs_filename}'
    LOADED_CANDIDATE_DICT = f'loaded_{LOCAL_TEST_INSTANCE}'
    
    loaded_test_instance = download_blob(
        bucket_name=train_output_gcs_bucket,
        source_gcs_obj=GCS_PATH_TO_BLOB,
        local_filename=LOADED_CANDIDATE_DICT
    )
    logging.info(f'loaded_test_instance: {loaded_test_instance}')
    
    # make prediction request
    _endpoint_id = _endpoint_uri.split('/')[-1]
    logging.info(f"_endpoint_id created = {_endpoint_id}")
    prediction_test = predict_custom_trained_model_sample(
        project=project,                     
        endpoint_id=_endpoint_id,
        location="us-central1",
        instances=loaded_test_instance
    )
    
    # ===================================================
    # Matching Engine
    # ===================================================
    logging.info(f"ann_index_endpoint_resource_uri: {ann_index_endpoint_resource_uri}")
    logging.info(f"brute_index_endpoint_resource_uri: {brute_index_endpoint_resource_uri}")

    deployed_ann_index = vertex_ai.MatchingEngineIndexEndpoint(ann_index_endpoint_resource_uri)
    deployed_bf_index = vertex_ai.MatchingEngineIndexEndpoint(brute_index_endpoint_resource_uri)

    DEPLOYED_ANN_ID = deployed_ann_index.deployed_indexes[0].id
    DEPLOYED_BF_ID = deployed_bf_index.deployed_indexes[0].id
    logging.info(f"DEPLOYED_ANN_ID: {DEPLOYED_ANN_ID}")
    logging.info(f"DEPLOYED_BF_ID: {DEPLOYED_BF_ID}")
    
    logging.info('Retreiving neighbors from ANN index...')
    
    start = time.time()
    ANN_response = deployed_ann_index.match(
        deployed_index_id=DEPLOYED_ANN_ID,
        queries=prediction_test,
        num_neighbors=10
    )
    elapsed_ann_time = time.time() - start
    elapsed_ann_time = round(elapsed_ann_time, 4)
    logging.info(f'ANN latency: {elapsed_ann_time} seconds')
    
    logging.info('Retreiving neighbors from BF index...')
    
    start = time.time()
    BF_response = deployed_bf_index.match(
        deployed_index_id=DEPLOYED_BF_ID,
        queries=prediction_test,
        num_neighbors=10
    )
    
    elapsed_bf_time = time.time() - start
    elapsed_bf_time = round(elapsed_bf_time, 4)
    logging.info(f'Bruteforce latency: {elapsed_bf_time} seconds')
    
    # =========================================================
    # Calculate recall by determining how many neighbors 
    # correctly retrieved as compared to the brute-force option
    # =========================================================
    recalled_neighbors = 0
    for tree_ah_neighbors, brute_force_neighbors in zip(
        ANN_response, BF_response
    ):
        tree_ah_neighbor_ids = [neighbor.id for neighbor in tree_ah_neighbors]
        brute_force_neighbor_ids = [neighbor.id for neighbor in brute_force_neighbors]

        recalled_neighbors += len(
            set(tree_ah_neighbor_ids).intersection(brute_force_neighbor_ids)
        )

    recall = recalled_neighbors / len(
        [neighbor for neighbors in BF_response for neighbor in neighbors]
    )
    
    # =========================================================
    # Metrics
    # =========================================================
    reduction = (elapsed_bf_time - elapsed_ann_time) / elapsed_bf_time*100.00
    increase  = (elapsed_bf_time - elapsed_ann_time)/elapsed_ann_time*100.00
    faster    = elapsed_bf_time / elapsed_ann_time

    logging.info(f"reduction in time         : {round(reduction, 3)}%")
    logging.info(f"% increase in performance : {round(increase, 3)}%")
    logging.info(f"how many times faster     : {round(faster, 3)}x faster")

    logging.info("Recall: {}".format(recall * 100.0))
    
    metrics.log_metric("Recall", (recall * 100.0))
    # metrics.log_metric("elapsed_query_time", elapsed_query_time)
    metrics.log_metric("elapsed_ann_time", elapsed_ann_time)
    metrics.log_metric("elapsed_bf_time", elapsed_bf_time)
    metrics.log_metric("latency_reduction", reduction)
    metrics.log_metric("perf_increase", increase)
    metrics.log_metric("x_faster", faster)
