from datetime import datetime
import time
import logging
from google.cloud import aiplatform_v1 as aipv1
from google.cloud.aiplatform_v1 import CreateIndexEndpointRequest
from google.cloud.aiplatform_v1.types.index import Index
from google.cloud.aiplatform_v1.types.index_endpoint import IndexEndpoint
from google.cloud.aiplatform_v1.types.index_endpoint import DeployedIndex

from . import helpers as helpers
# from helpers import _build_index_config, ResourceNotExistException

from google.protobuf import struct_pb2
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class MatchingEngineCRUD:
    def __init__(
        self
        , project_id: str
        , region: str
        , project_num: int
        , index_type: str
        , index_name: str = None
        , vpc_network_name: str = None
    ):
        self.project_id = project_id
        self.project_num = project_num
        self.region = region
        self.index_name = index_name if index_name is not None else None
        self.vpc_network_name = vpc_network_name if vpc_network_name is not None else None
        self.index_type = index_type
        
        self.index_endpoint_name = f"{self.index_name}_endpoint" if self.index_name is not None else None
        self.PARENT = f"projects/{self.project_num}/locations/{self.region}"

        ENDPOINT = f"{self.region}-aiplatform.googleapis.com"
        
        # set index client
        self.index_client = aipv1.IndexServiceClient(
            client_options=dict(api_endpoint=ENDPOINT)
        )
        # set index endpoint client
        self.index_endpoint_client = aipv1.IndexEndpointServiceClient(
            client_options=dict(api_endpoint=ENDPOINT)
        )

    def _set_index_name(self, index_name: str) -> None:
        """

        :param index_name:
        :return:
        """
        self.index_name = index_name

    def _set_index_endpoint_name(self, index_endpoint_name: str = None) -> None:
        """

        :param index_endpoint_name:
        :return:
        """
        if index_endpoint_name is not None:
            self.index_endpoint_name = index_endpoint_name
        elif self.index_name is not None:
            self.index_endpoint_name = f"{self.index_name}_endpoint"
        else:
            raise ResourceNotExistException("index")

    def _get_index(self) -> Index:
        """

        :return:
        """
        # Check if index exists
        if self.index_name is not None:
            indexes = [
                index.name for index in self.list_indexes()
                if index.display_name == self.index_name
            ]
        else:
            raise ResourceNotExistException("index")

        if len(indexes) == 0:
            return None
        else:
            index_id = indexes[0]
            request = aipv1.GetIndexRequest(name=index_id)
            index = self.index_client.get_index(request=request)
            return index

    def _get_index_endpoint(self) -> IndexEndpoint:
        """

        :return:
        """
        # Check if index endpoint exists
        if self.index_endpoint_name is not None:
            index_endpoints = [
                response.name for response in self.list_index_endpoints()
                if response.display_name == self.index_endpoint_name
            ]
        else:
            raise ResourceNotExistException("index_endpoint")

        if len(index_endpoints) == 0:
            logging.info(f"Could not find index endpoint: {self.index_endpoint_name}")
            return None
        else:
            index_endpoint_id = index_endpoints[0]
            index_endpoint = self.index_endpoint_client.get_index_endpoint(
                name=index_endpoint_id
            )
            return index_endpoint

    def list_indexes(self) -> List[Index]:
        """

        :return:
        """
        request = aipv1.ListIndexesRequest(parent=self.PARENT)
        page_result = self.index_client.list_indexes(request=request)
        indexes = [
            response for response in page_result
        ]
        return indexes

    def list_index_endpoints(self) -> List[IndexEndpoint]:
        """

        :return:
        """
        request = aipv1.ListIndexEndpointsRequest(parent=self.PARENT)
        page_result = self.index_endpoint_client.list_index_endpoints(request=request)
        index_endpoints = [
            response for response in page_result
        ]
        return index_endpoints

    def list_deployed_indexes(
        self
        , endpoint_name: str = None
    ) -> List[DeployedIndex]:
        """

        :param endpoint_name:
        :return:
        """
        try:
            if endpoint_name is not None:
                self._set_index_endpoint_name(endpoint_name)
            index_endpoint = self._get_index_endpoint()
            deployed_indexes = index_endpoint.deployed_indexes
        except ResourceNotExistException as rnee:
            raise rnee

        return list(deployed_indexes)

    def create_index(
        self
        , embedding_gcs_uri: str
        , dimensions: int
        , index_name: str = None
    ) -> Index:
        """

        :param index_name:
        :param embedding_gcs_uri:
        :param dimensions:
        :return:
        """
        if index_name is not None:
            self._set_index_name(index_name)
        # Get index
        if self.index_name is None:
            raise ResourceNotExistException("index")
        index = self._get_index()
        # Create index if does not exists
        if index:
            logger.info(f"Index {self.index_name} already exists with id {index.name}")
        else:
            logger.info(f"Index {self.index_name} does not exists. Creating index ...")

            metadata = helpers._build_index_config(
                embedding_gcs_uri=embedding_gcs_uri
                , dimensions=dimensions
                , index_type=self.index_type
            )

            index_request = {
                "display_name": self.index_name,
                "description": "Index for LangChain demo",
                "metadata": struct_pb2.Value(struct_value=metadata),
                "index_update_method": aipv1.Index.IndexUpdateMethod.STREAM_UPDATE,
            }

            r = self.index_client.create_index(
                parent=self.PARENT,
                index=Index(index_request)
            )

            # Poll the operation until it's done successfully.
            logging.info("Poll the operation to create index ...")
            while True:
                if r.done():
                    break
                time.sleep(60)
                print('.', end='')

            index = r.result()
            logger.info(f"Index {self.index_name} created with resource name as {index.name}")

        return index

    # TODO: this is generating an error about publicEndpointEnabled not being set without network
    def create_index_endpoint(
        self
        , endpoint_name: str = None
        , network: str = None
    ) -> IndexEndpoint:
        """

        :param endpoint_name:
        :param network:
        :return:
        """
        try:
            if endpoint_name is not None:
                self._set_index_endpoint_name(endpoint_name)
            # Get index endpoint if exists
            index_endpoint = self._get_index_endpoint()

            # Create Index Endpoint if does not exists
            if index_endpoint is not None:
                logger.info("Index endpoint already exists")
            else:
                logger.info(f"Index endpoint {self.index_endpoint_name} does not exists. Creating index endpoint...")
                index_endpoint_request = {
                    "display_name": self.index_endpoint_name
                }
                index_endpoint = IndexEndpoint(index_endpoint_request)
                if network is not None:
                    index_endpoint.network = network
                else:
                    index_endpoint.public_endpoint_enabled = True
                    index_endpoint.publicEndpointEnabled = True
                r = self.index_endpoint_client.create_index_endpoint(
                        parent=self.PARENT,
                        index_endpoint=index_endpoint
                )

                logger.info("Poll the operation to create index endpoint ...")
                while True:
                    if r.done():
                        break
                    time.sleep(60)
                    print('.', end='')

                index_endpoint = r.result()
        except Exception as e:
            logger.error(f"Failed to create index endpoint {self.index_endpoint_name}")
            raise e

        return index_endpoint

    def deploy_index(
        self
        , index_name: str = None
        , endpoint_name: str = None
        , machine_type: str = "e2-standard-2"
        , min_replica_count: int = 2
        , max_replica_count: int = 2
    ) -> IndexEndpoint:
        """

        :param endpoint_name:
        :param index_name:
        :param machine_type:
        :param min_replica_count:
        :param max_replica_count:
        :return:
        """
        if index_name is not None:
            self._set_index_name(index_name)
        if endpoint_name is not None:
            self._set_index_endpoint_name(endpoint_name)

        index = self._get_index()
        index_endpoint = self._get_index_endpoint()
        # Deploy Index to endpoint
        try:
            # Check if index is already deployed to the endpoint
            if index.name in index_endpoint.deployed_indexes:
                logger.info(f"Skipping deploying Index. Index {self.index_name}" +
                            f"already deployed with id {index.name} to the index endpoint {self.index_endpoint_name}")
                return index_endpoint

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            deployed_index_id = f"{self.index_name.replace('-', '_')}_{timestamp}"
            deploy_index = {
                "id": deployed_index_id,
                "display_name": deployed_index_id,
                "index": index.name,
                "dedicated_resources": {
                    "machine_spec": {
                        "machine_type": machine_type,
                        },
                    "min_replica_count": min_replica_count,
                    "max_replica_count": max_replica_count
                }
            }
            logger.info(f"Deploying index with request = {deploy_index}")
            r = self.index_endpoint_client.deploy_index(
                index_endpoint=index_endpoint.name,
                deployed_index=DeployedIndex(deploy_index)
            )

            # Poll the operation until it's done successfullly.
            logger.info("Poll the operation to deploy index ...")
            while True:
                if r.done():
                    break
                time.sleep(60)
                print('.', end='')

            logger.info(f"Deployed index {self.index_name} to endpoint {self.index_endpoint_name}")

        except Exception as e:
            logger.error(f"Failed to deploy index {self.index_name} to the index endpoint {self.index_endpoint_name}")
            raise e

        return index_endpoint

    def get_index_and_endpoint(self) -> (str, str):
        """

        :return:
        """
        # Get index id if exists
        index = self._get_index()
        index_id = index.name if index else ''

        # Get index endpoint id if exists
        index_endpoint = self._get_index_endpoint()
        index_endpoint_id = index_endpoint.name if index_endpoint else ''

        return index_id, index_endpoint_id

    def delete_index(
        self
        , index_name: str = None
    ) -> str:
        """
        :param index_name: str
        :return:
        """
        if index_name is not None:
            self._set_index_name(index_name)
        # Check if index exists
        index = self._get_index()

        # create index if does not exists
        if index:
            # Delete index
            index_id = index.name
            logger.info(f"Deleting Index {self.index_name} with id {index_id}")
            self.index_client.delete_index(name=index_id)
            return f"index {index_id} deleted."
        else:
            raise ResourceNotExistException(f"{self.index_name}")

    def undeploy_index(
        self
        , index_name: str
        , endpoint_name: str
    ):
        """

        :param index_name:
        :param endpoint_name:
        :return:
        """
        logger.info(f"Undeploying index with id {index_name} from Index endpoint {endpoint_name}")
        endpoint_id = f"{self.PARENT}/indexEndpoints/{endpoint_name}"
        r = self.index_endpoint_client.undeploy_index(
            index_endpoint=endpoint_id
            , deployed_index_id=index_name
        )
        response = r.result()
        logger.info(response)
        return response.display_name

    def delete_index_endpoint(
        self
        , index_endpoint_name: str = None
    ) -> str:
        """

        :param index_endpoint_name: str
        :return:
        """
        if index_endpoint_name is not None:
            self._set_index_endpoint_name(index_endpoint_name)
        # Check if index endpoint exists
        index_endpoint = self._get_index_endpoint()

        # Create Index Endpoint if does not exists
        if index_endpoint is not None:
            logger.info(
                f"Index endpoint {self.index_endpoint_name}  exists with resource " +
                f"name as {index_endpoint.name}" #+
                # f"{index_endpoint.public_endpoint_domain_name}")
            )

            #index_endpoint_id = index_endpoint.name
            #index_endpoint = self.index_endpoint_client.get_index_endpoint(
            #    name=index_endpoint.name
            #)
            
            # Undeploy existing indexes
            for d_index in index_endpoint.deployed_indexes:
                self.undeploy_index(
                    index_name=d_index.id
                    , endpoint_name=index_endpoint_name
                )

            # Delete index endpoint
            logger.info(f"Deleting Index endpoint {self.index_endpoint_name} with id {index_endpoint_id}")
            self.index_endpoint_client.delete_index_endpoint(name=index_endpoint.name)
            return f"Index endpoint {index_endpoint.name} deleted."
        else:
            raise ResourceNotExistException(f"{self.index_endpoint_name}")