
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (
    Artifact, Dataset, Input, InputPath, 
    Model, Output, OutputPath, component, Metrics
)
@kfp.v2.dsl.component(
    base_image='python:3.9',
    packages_to_install=[
        'google-cloud-aiplatform==1.26.1',
        # 'tensorflow==2.9.2',
        # 'tensorflow-recommenders==0.7.0',
        'numpy',
        'google-cloud-storage',
    ],
    # output_component_file="./pipelines/train_custom_model.yaml",
)
def train_custom_model(
    project: str,
    location: str,
    model_version: str,
    pipeline_version: str,
    model_name: str, 
    worker_pool_specs: dict,
    # vocab_dict_uri: str, 
    train_output_gcs_bucket: str,                         # change to workdir?
    training_image_uri: str,
    tensorboard_resource_name: str,
    service_account: str,
    experiment_name: str,
    experiment_run: str,
    generate_new_vocab: bool,
) -> NamedTuple('Outputs', [
    ('job_dict_uri', str),
    ('query_tower_dir_uri', str),
    ('candidate_tower_dir_uri', str),
    ('experiment_run_dir', str),
]):
    
    import logging
    import numpy as np
    import pickle as pkl
    
    from google.cloud import aiplatform as vertex_ai
    # import google.cloud.aiplatform_v1beta1 as aip_beta
    from google.cloud import storage
    
    vertex_ai.init(
        project=project,
        location=location,
        experiment=experiment_name,
    )
    
    storage_client = storage.Client(project=project)
    
    JOB_NAME = f'train-{model_name}'
    logging.info(f'JOB_NAME: {JOB_NAME}')
    
    BASE_OUTPUT_DIR = f'gs://{train_output_gcs_bucket}/{experiment_name}/{experiment_run}'
    logging.info(f'BASE_OUTPUT_DIR: {BASE_OUTPUT_DIR}')
    
    # logging.info(f'vocab_dict_uri: {vocab_dict_uri}')
    
    logging.info(f'tensorboard_resource_name: {tensorboard_resource_name}')
    logging.info(f'service_account: {service_account}')
    logging.info(f'worker_pool_specs: {worker_pool_specs}')
    
    # ====================================================
    # Launch Vertex job
    # ====================================================
    
    worker_pool_specs[0]['container_spec']['args'].append(f'--tb_resource_name={tensorboard_resource_name}')
    
    if generate_new_vocab == 'True':
        worker_pool_specs[0]['container_spec']['args'].append(f'--new_vocab')
  
    job = vertex_ai.CustomJob(
        display_name=JOB_NAME,
        worker_pool_specs=worker_pool_specs,
        base_output_dir=BASE_OUTPUT_DIR,
        staging_bucket=f"{BASE_OUTPUT_DIR}/staging",
    )
    
    logging.info(f'Submitting train job to Vertex AI...')
    
    job.run(
        tensorboard=tensorboard_resource_name,
        service_account=f'{service_account}',
        restart_job_on_worker_restart=False,
        enable_web_access=True,
        sync=False,
    )
        
    # wait for job to complete
    job.wait()
    
    # ====================================================
    # Save job details
    # ====================================================
    
    train_job_dict = job.to_dict()
    logging.info(f'train_job_dict: {train_job_dict}')
    
    # pkl dict to GCS
    logging.info(f"Write pickled dict to GCS...")
    TRAIN_DICT_LOCAL = f'train_job_dict.pkl'
    TRAIN_DICT_GCS_OBJ = f'{experiment_name}/{experiment_run}/{TRAIN_DICT_LOCAL}' # destination folder prefix and blob name
    
    logging.info(f"TRAIN_DICT_LOCAL: {TRAIN_DICT_LOCAL}")
    logging.info(f"TRAIN_DICT_GCS_OBJ: {TRAIN_DICT_GCS_OBJ}")

    # pickle
    filehandler = open(f'{TRAIN_DICT_LOCAL}', 'wb')
    pkl.dump(train_job_dict, filehandler)
    filehandler.close()
    
    # upload to GCS
    bucket_client = storage_client.bucket(train_output_gcs_bucket)
    blob = bucket_client.blob(TRAIN_DICT_GCS_OBJ)
    blob.upload_from_filename(TRAIN_DICT_LOCAL)
    
    job_dict_uri = f'gs://{train_output_gcs_bucket}/{TRAIN_DICT_GCS_OBJ}'
    logging.info(f"{TRAIN_DICT_LOCAL} uploaded to {job_dict_uri}")
    
    # ====================================================
    # Model and index artifact uris
    # ====================================================
    EXPERIMENT_RUN_DIR = f"gs://{train_output_gcs_bucket}/{experiment_name}/{experiment_run}"
    query_tower_dir_uri = f"{EXPERIMENT_RUN_DIR}/model-dir/query_model" 
    candidate_tower_dir_uri = f"{EXPERIMENT_RUN_DIR}/model-dir/candidate_model"
    # candidate_index_dir_uri = f"gs://{output_dir_gcs_bucket_name}/{experiment_name}/{experiment_run}/candidate_model"
    
    logging.info(f'query_tower_dir_uri: {query_tower_dir_uri}')
    logging.info(f'candidate_tower_dir_uri: {candidate_tower_dir_uri}')
    # logging.info(f'candidate_index_dir_uri: {candidate_index_dir_uri}')
    
    return (
        f'{job_dict_uri}',
        f'{query_tower_dir_uri}',
        f'{candidate_tower_dir_uri}',
        f'{EXPERIMENT_RUN_DIR}',
    )
