
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
def model_monitoring_config(
    project: str,
    location: str,
    version: str,
    prefix: str,
    emails: str,
    train_output_gcs_bucket: str,
    # feature_dict: dict, # TODO
    bq_dataset: str,
    bq_train_table: str,
    experiment_name: str,
    experiment_run: str,
    endpoint: str,
):
    # TODO - imports
    
    import logging
    from datetime import datetime
    import time
    import numpy as np
    import pickle as pkl
    
    logging.getLogger().setLevel(logging.INFO)
    # google cloud SDKs
    from google.cloud import storage
    from google.cloud import aiplatform as vertex_ai
    from google.cloud.aiplatform import model_monitoring
    
    from google.cloud import storage
    from google.cloud.storage.bucket import Bucket
    from google.cloud.storage.blob import Blob
    
    from google.protobuf import json_format
    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Value
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources
    
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
    
    
    USER_EMAILS = [emails]
    alert_config = model_monitoring.EmailAlertConfig(USER_EMAILS, enable_logging=True)
    
    MONITOR_INTERVAL = 1
    schedule_config = model_monitoring.ScheduleConfig(monitor_interval=MONITOR_INTERVAL)
    
    SAMPLE_RATE = 0.8

    logging_sampling_strategy = model_monitoring.RandomSampleConfig(sample_rate=SAMPLE_RATE)
    
    # ===================================================
    # feature dict
    # ===================================================
    QUERY_FILENAME = 'query_feats_dict.pkl'
    # FEATURES_PREFIX = f'{experiment_name}/{experiment_run}/features'
    GCS_PATH_TO_BLOB = f'{experiment_name}/{experiment_run}/features/{QUERY_FILENAME}'
    
    loaded_feat_dict = download_blob(
        bucket_name=train_output_gcs_bucket,
        source_gcs_obj=GCS_PATH_TO_BLOB,
        local_filename=QUERY_FILENAME
    )
    logging.info(f'loaded_feat_dict: {loaded_feat_dict}')
    
    filehandler = open(QUERY_FILENAME, 'rb')
    FEAT_DICT = pkl.load(filehandler)
    filehandler.close()

    
    feature_names = list(FEAT_DICT.keys())

    # =========================== #
    ##   Feature value drift     ##
    # =========================== #
    DRIFT_THRESHOLD_VALUE = 0.05
    ATTRIBUTION_DRIFT_THRESHOLD_VALUE = 0.05
    
    drift_thresholds = dict()

    for feature in feature_names:
        if feature in drift_thresholds:
            print("feature name already in dict")
        else:
            drift_thresholds[feature] = DRIFT_THRESHOLD_VALUE

    logging.info(f"drift_thresholds      : {drift_thresholds}\n")
    
    drift_config = model_monitoring.DriftDetectionConfig(
        drift_thresholds=drift_thresholds,
        # attribute_drift_thresholds=attr_drift_thresholds,
    )

    # =========================== #
    ##   Feature value skew      ##
    # =========================== #
    TRAIN_DATA_SOURCE_URI = f"bq://{project}.{bq_dataset}.{bq_train_table}"
    logging.info(f"TRAIN_DATA_SOURCE_URI = {TRAIN_DATA_SOURCE_URI}")
    
    SKEW_THRESHOLD_VALUE = 0.05
    ATTRIBUTION_SKEW_THRESHOLD_VALUE = 0.05
    
    skew_thresholds = dict()

    for feature in feature_names:
        if feature in skew_thresholds:
            logging.info("feature name already in dict")
        else:
            skew_thresholds[feature] = SKEW_THRESHOLD_VALUE        
    logging.info(f"skew_thresholds      : {skew_thresholds}\n")
    
    # skew config
    skew_config = model_monitoring.SkewDetectionConfig(
        data_source=TRAIN_DATA_SOURCE_URI,
        # data_format = TRAIN_DATA_FORMAT, # only used if source in GCS
        skew_thresholds=skew_thresholds,
        # attribute_skew_thresholds=attribute_skew_thresholds,
        # target_field=TARGET, # no target; embedding model
    )
    
    # ====================================================
    # objective_config
    # ====================================================
    objective_config = model_monitoring.ObjectiveConfig(
        skew_detection_config=skew_config,
        drift_detection_config=drift_config,
        explanation_config=None,
    )
    
    # ====================================================
    # launch monitoring_job
    # ====================================================
    
    JOB_DISPLAY_NAME = f"mm_pipe_{experiment_run}_{prefix}"
    logging.info(f"JOB_DISPLAY_NAME: {JOB_DISPLAY_NAME}")

    monitoring_job = vertex_ai.ModelDeploymentMonitoringJob.create(
        display_name=JOB_DISPLAY_NAME,
        project=project,
        location=location,
        endpoint=_endpoint,
        logging_sampling_strategy=logging_sampling_strategy,
        schedule_config=schedule_config,
        alert_config=alert_config,
        objective_configs=objective_config,
    )
    
    logging.info(f"monitoring_job: {monitoring_job.resource_name}")
