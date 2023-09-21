from decouple import config
import time
import os
from enum import Enum
from google.protobuf import struct_pb2

def _get_api_key() -> str:
    """
    Retrieve API key from Secret Manager

    :return: str
    """
    sm_client = secretmanager.SecretManagerServiceClient()
    name = sm_client.secret_path(project, _SECRET_ID)
    response = sm_client.access_secret_version(request={"name": _SECRET_VERSION})

    return response.payload.data.decode("UTF-8")
    

def _build_index_config(embedding_gcs_uri: str, dimensions: int, index_type: str):
    
    """
    index_type: should either be "ann" or "bf"
    
    """
    
    if index_type == "ann":
        _treeAhConfig = struct_pb2.Struct(
            fields={
                "leafNodeEmbeddingCount": struct_pb2.Value(number_value=500),
                "leafNodesToSearchPercent": struct_pb2.Value(number_value=7),
            }
        )
        _algorithmConfig = struct_pb2.Struct(
            fields={"treeAhConfig": struct_pb2.Value(struct_value=_treeAhConfig)}
        )
        _config = struct_pb2.Struct(
            fields={
                "dimensions": struct_pb2.Value(number_value=dimensions),
                "approximateNeighborsCount": struct_pb2.Value(number_value=150),
                "distanceMeasureType": struct_pb2.Value(string_value="DOT_PRODUCT_DISTANCE"),
                "algorithmConfig": struct_pb2.Value(struct_value=_algorithmConfig),
                "shardSize": struct_pb2.Value(string_value="SHARD_SIZE_SMALL"), # TODO - parametrize
            }
        )
        metadata = struct_pb2.Struct(
            fields={
                "config": struct_pb2.Value(struct_value=_config),
                "contentsDeltaUri": struct_pb2.Value(string_value=embedding_gcs_uri),
            }
        )
    elif index_type == "bf":

        _algorithmConfig = struct_pb2.Struct(
            fields={"bruteForceConfig": struct_pb2.Value(struct_value=struct_pb2.Struct())}
        )
        _config = struct_pb2.Struct(
            fields={
                "dimensions": struct_pb2.Value(number_value=dimensions),
                "approximateNeighborsCount": struct_pb2.Value(number_value=150),
                "distanceMeasureType": struct_pb2.Value(string_value="DOT_PRODUCT_DISTANCE"),
                "algorithmConfig": struct_pb2.Value(struct_value=_algorithmConfig),
                "shardSize": struct_pb2.Value(string_value="SHARD_SIZE_SMALL"), # TODO - parametrize
            }
        )
        metadata = struct_pb2.Struct(
            fields={
                "config": struct_pb2.Value(struct_value=_config),
                "contentsDeltaUri": struct_pb2.Value(string_value=embedding_gcs_uri),
            }
        )
        
    else:
        logger.error(f'Failed to create index. select either "ann" or "bf" for "index_type"')
        
        # TODO - add try / except logic:
        # except Exception as e:
        #     logger.error(f'Failed to create index. select either "ann" or "bf" for "index_type"')
        #     raise e

    return metadata

class ResourceNotExistException(Exception):
    def __init__(self, resource: str, message="Resource Does Not Exist."):
        self.resource = resource
        self.message = message
        super().__init__(self.message)