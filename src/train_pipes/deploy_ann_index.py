
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
        'google-api-core==2.11.0',
    ]
)
def deploy_ann_index(
    project: str,
    location: str,
    version: str,
    deployed_ann_index_name: str,
    ann_index_resource_uri: str,
    index_endpoint_resource_uri: str,
) -> NamedTuple('Outputs', [
    ('index_endpoint_resource_uri', str),
    ('ann_index_resource_uri', str),
    ('deployed_ann_index_name', str),
    ('deployed_ann_index', Artifact),
]):
  
    import logging
    from google.cloud import aiplatform as vertex_ai
    from datetime import datetime
    import time

    vertex_ai.init(
        project=project,
        location=location,
    )
    
    # define vars
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    # deployed_ann_index_name = deployed_ann_index_name.replace('-', '_')
    # logging.info(f"deployed_ann_index_name: {deployed_ann_index_name}")
    
    DEPLOYED_INDEX_NAME = f'{deployed_ann_index_name}-{TIMESTAMP}'.replace('-', '_')
    logging.info(f"DEPLOYED_INDEX_NAME: {DEPLOYED_INDEX_NAME}")
    
    # init index
    ann_index = vertex_ai.MatchingEngineIndex(
      index_name=ann_index_resource_uri
    )
    ann_index_resource_uri = ann_index.resource_name
    logging.info(f"ann_index_resource_uri: {ann_index_resource_uri}")

    # init index endpoint
    index_endpoint = vertex_ai.MatchingEngineIndexEndpoint(
      index_endpoint_resource_uri
    )
    logging.info(f"index_endpoint: {index_endpoint}")

    # deploy index to endpoint
    index_endpoint = index_endpoint.deploy_index(
      index=ann_index, 
      deployed_index_id=DEPLOYED_INDEX_NAME
    )

    logging.info(f"index_endpoint.deployed_indexes: {index_endpoint.deployed_indexes}")
    INDEX_ID = index_endpoint.deployed_indexes[0].id
    logging.info(f"INDEX_ID: {INDEX_ID}")

    return (
      f'{index_endpoint_resource_uri}',
      f'{ann_index_resource_uri}',
      f'{deployed_ann_index_name}',
      ann_index,
    )
