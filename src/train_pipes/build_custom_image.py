
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)
@kfp.v2.dsl.component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
    packages_to_install=[
        "google-cloud-build"
    ],
)
def build_custom_image(
    project: str,
    artifact_gcs_path: str,
    docker_name: str,
    app_dir_name: str,
    custom_image_uri: str,
) -> NamedTuple('Outputs', [
    ('custom_image_uri', str),
]):
    # TODO: make output Artifact for image_uri
    """
    custom pipeline component to build custom image using
    Cloud Build, the training/serving application code, and dependencies
    defined in the Dockerfile
    """
    
    import logging
    import os

    from google.cloud.devtools import cloudbuild_v1 as cloudbuild
    from google.protobuf.duration_pb2 import Duration

    # initialize client for cloud build
    logging.getLogger().setLevel(logging.INFO)
    build_client = cloudbuild.services.cloud_build.CloudBuildClient()
    
    # parse step inputs to get path to Dockerfile and training application code
    _gcs_dockerfile_path = os.path.join(artifact_gcs_path, f"{docker_name}") # Dockerfile.XXXXX
    _gcs_script_dir_path = os.path.join(artifact_gcs_path, f"{app_dir_name}/") # "trainer/"
    
    logging.info(f"_gcs_dockerfile_path: {_gcs_dockerfile_path}")
    logging.info(f"_gcs_script_dir_path: {_gcs_script_dir_path}")
    
    # define build steps to pull the training code and Dockerfile
    # and build/push the custom training container image
    build = cloudbuild.Build()
    build.steps = [
        {
            "name": "gcr.io/cloud-builders/gsutil",
            "args": ["cp", "-r", _gcs_script_dir_path, "."],
        },
        {
            "name": "gcr.io/cloud-builders/gsutil",
            "args": ["cp", _gcs_dockerfile_path, "Dockerfile"],
        },
        # enabling Kaniko cache in a Docker build that caches intermediate
        # layers and pushes image automatically to Container Registry
        # https://cloud.google.com/build/docs/kaniko-cache
        # {
        #     "name": "gcr.io/kaniko-project/executor:latest",
        #     # "name": "gcr.io/kaniko-project/executor:v1.8.0",        # TODO; downgraded to avoid error in build
        #     # "args": [f"--destination={training_image_uri}", "--cache=true"],
        #     "args": [f"--destination={training_image_uri}", "--cache=false"],
        # },
        {
            "name": "gcr.io/cloud-builders/docker",
            "args": ['build','-t', f'{custom_image_uri}', '.'],
        },
        {
            "name": "gcr.io/cloud-builders/docker",
            "args": ['push', f'{custom_image_uri}'], 
        },
    ]
    # override default timeout of 10min
    timeout = Duration()
    timeout.seconds = 7200
    build.timeout = timeout

    # create build
    operation = build_client.create_build(project_id=project, build=build)
    logging.info("IN PROGRESS:")
    logging.info(operation.metadata)

    # get build status
    result = operation.result()
    logging.info("RESULT:", result.status)

    # return step outputs
    return (
        custom_image_uri,
    )
