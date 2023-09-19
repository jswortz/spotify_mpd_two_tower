
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
        'google-api-core==2.11.0'
        # 'google-cloud-storage',
    ],
)
def create_ann_index(
    project: str,
    location: str,
    version: str, 
    vpc_network_name: str,
    emb_index_gcs_uri: str,
    dimensions: int,
    ann_index_display_name: str,
    approximate_neighbors_count: int,
    distance_measure_type: str,
    leaf_node_embedding_count: int,
    leaf_nodes_to_search_percent: int, 
    ann_index_description: str,
    # ann_index_labels: Dict, 
) -> NamedTuple('Outputs', [
    ('ann_index_resource_uri', str),
    ('ann_index', Artifact),
]):
    import logging
    from google.cloud import aiplatform as vertex_ai
    from datetime import datetime
    import time

    vertex_ai.init(
        project=project,
        location=location,
    )
    
    VERSION = version.replace('_', '-')
    
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
    
    ENDPOINT = "{}-aiplatform.googleapis.com".format(location)
    NETWORK_NAME = vpc_network_name
    INDEX_DIR_GCS = emb_index_gcs_uri
    PARENT = "projects/{}/locations/{}".format(project, location)

    logging.info(f"ENDPOINT: {ENDPOINT}")
    logging.info(f"project: {project}")
    logging.info(f"location: {location}")
    logging.info(f"INDEX_DIR_GCS: {INDEX_DIR_GCS}")
    
    display_name = f'{ann_index_display_name}-{VERSION}'
    
    logging.info(f"display_name: {display_name}")
    
    # ==============================================================================
    # Create Index 
    # ==============================================================================

    start = time.time()
        
    tree_ah_index = vertex_ai.MatchingEngineIndex.create_tree_ah_index(
        display_name=display_name,
        contents_delta_uri=f'{emb_index_gcs_uri}', # emb_index_gcs_uri,
        dimensions=dimensions,
        approximate_neighbors_count=approximate_neighbors_count,
        distance_measure_type=distance_measure_type,
        leaf_node_embedding_count=leaf_node_embedding_count,
        leaf_nodes_to_search_percent=leaf_nodes_to_search_percent,
        description=ann_index_description,
        # labels=ann_index_labels,
        sync=True,
    )

    end = time.time()
    elapsed_time = round((end - start), 2)
    logging.info(f'Elapsed time creating index: {elapsed_time} seconds\n')
    
    ann_index_resource_uri = tree_ah_index.resource_name
    logging.info("ann_index_resource_uri:", ann_index_resource_uri) 

    return (
      f'{ann_index_resource_uri}',
      tree_ah_index,
    )
