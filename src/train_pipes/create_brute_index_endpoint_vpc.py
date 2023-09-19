
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
    ],
)
def create_brute_index_endpoint_vpc(
    bf_index_artifact: Input[Artifact],
    project: str,
    project_number: str,
    location: str,
    version: str,
    vpc_network_name: str,
    brute_index_endpoint_display_name: str,
    brute_index_endpoint_description: str,
    brute_force_index_resource_uri: str,
) -> NamedTuple('Outputs', [
    ('vpc_network_resource_uri', str),
    ('brute_index_endpoint_resource_uri', str),
    ('brute_index_endpoint', Artifact),
    ('brute_index_endpoint_display_name', str),
    ('brute_force_index_resource_uri', str),
]):

    import logging
    from google.cloud import aiplatform as vertex_ai
    from datetime import datetime
    import time

    vertex_ai.init(
        project=project,
        location=location,
    )
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    vpc_network_resource_uri = f'projects/{project_number}/global/networks/{vpc_network_name}'
    logging.info(f"vpc_network_resource_uri: {vpc_network_resource_uri}")

    brute_index_endpoint = vertex_ai.MatchingEngineIndexEndpoint.create(
        display_name=f'{brute_index_endpoint_display_name}',
        description=brute_index_endpoint_description,
        network=vpc_network_resource_uri,
    )
    brute_index_endpoint_resource_uri = brute_index_endpoint.resource_name
    logging.info(f"brute_index_endpoint_resource_uri: {brute_index_endpoint_resource_uri}")

    return (
      f'{vpc_network_resource_uri}',
      f'{brute_index_endpoint_resource_uri}',
      brute_index_endpoint,
      f'{brute_index_endpoint_display_name}',
      f'{brute_force_index_resource_uri}',
    )
