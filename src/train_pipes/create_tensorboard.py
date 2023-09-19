
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
        'numpy',
        'google-cloud-storage',
    ],
    # output_component_file="./pipelines/train_custom_model.yaml",
)
def create_tensorboard(
    project: str,
    location: str,
    model_version: str,
    pipeline_version: str,
    model_name: str, 
    experiment_name: str,
    experiment_run: str,
) -> NamedTuple('Outputs', [
    ('tensorboard_resource_name', str),
    ('tensorboard_display_name', str),
]):
    
    import logging
    from google.cloud import aiplatform as vertex_ai
    from google.cloud import storage
    
    vertex_ai.init(
        project=project,
        location=location,
        # experiment=experiment_name,
    )
    
    logging.info(f'experiment_name: {experiment_name}')
    
    # # create new TB instance
    TENSORBOARD_DISPLAY_NAME=f"{experiment_name}-v1"
    tensorboard = vertex_ai.Tensorboard.create(display_name=TENSORBOARD_DISPLAY_NAME, project=project, location=location)
    TB_RESOURCE_NAME = tensorboard.resource_name
    
    logging.info(f'TENSORBOARD_DISPLAY_NAME: {TENSORBOARD_DISPLAY_NAME}')
    logging.info(f'TB_RESOURCE_NAME: {TB_RESOURCE_NAME}')
    
    return (
        f'{TB_RESOURCE_NAME}',
        f'{TENSORBOARD_DISPLAY_NAME}',
    )
