
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
        # 'google-cloud-storage',
    ],
)
def create_brute_force_index(
    project: str,
    location: str,
    version: str,
    vpc_network_name: str,
    emb_index_gcs_uri: str,
    dimensions: int,
    brute_force_index_display_name: str,
    approximate_neighbors_count: int,
    distance_measure_type: str,
    brute_force_index_description: str,
    # brute_force_index_labels: Dict,
) -> NamedTuple('Outputs', [
    ('brute_force_index_resource_uri', str),
    ('brute_force_index', Artifact),
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
    VERSION = version.replace('_', '-')
    
    ENDPOINT = "{}-aiplatform.googleapis.com".format(location)
    NETWORK_NAME = vpc_network_name
    INDEX_DIR_GCS = emb_index_gcs_uri
    PARENT = "projects/{}/locations/{}".format(project, location)

    logging.info("ENDPOINT: {}".format(ENDPOINT))
    logging.info("PROJECT_ID: {}".format(project))
    logging.info("REGION: {}".format(location))
    
    display_name = f'{brute_force_index_display_name}_{VERSION}'
    
    logging.info(f"display_name: {display_name}")
    
    # ==============================================================================
    # Create Index 
    # ==============================================================================

    start = time.time()
    
    brute_force_index = vertex_ai.MatchingEngineIndex.create_brute_force_index(
        display_name=display_name,
        contents_delta_uri=f'{emb_index_gcs_uri}', # emb_index_gcs_uri,
        dimensions=dimensions,
        # approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=distance_measure_type,
        description=brute_force_index_description,
        # labels=brute_force_index_labels,
        sync=True,
    )
    brute_force_index_resource_uri = brute_force_index.resource_name
    print("brute_force_index_resource_uri:",brute_force_index_resource_uri) 

    return (
      f'{brute_force_index_resource_uri}',
      brute_force_index,
    )
