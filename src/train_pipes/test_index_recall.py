
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'google-cloud-aiplatform==1.20.0',
        # 'google-cloud-storage',
    ],
)
def test_index_recall(
    project: str,
    location: str,
    version: str,
    ann_index_resource_uri: str,
    brute_force_index_resource_uri: str,
    endpoint: Input[Artifact],
    metrics: Output[Metrics],
):
    # here
    
    import base64
    import logging

    from typing import Dict, List, Union

    from google.cloud import aiplatform as vertex_ai
    from google.protobuf import json_format
    from google.protobuf.json_format import Parse
    from google.protobuf.struct_pb2 import Value
    
    from google_cloud_pipeline_components.proto.gcp_resources_pb2 import GcpResources

    logging.getLogger().setLevel(logging.INFO)
    vertex_ai.init(
        project=project,
        location=location,
    )
    
    endpoint_resource_path = endpoint.metadata["resourceName"]

    # define endpoint resource in component
    logging.info(f"endpoint_resource_path = {endpoint_resource_path}")
    _endpoint = vertex_ai.Endpoint(endpoint_resource_path)
    
    ################################################################################
    # Helper function for returning endpoint predictions via required json format
    ################################################################################

    def predict_custom_trained_model_sample(
        project: str,
        endpoint_id: str,
        instances: Dict,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
        """
        `instances` can be either single instance of type dict or a list
        of instances.
        """

        ########################################################################
        # Initialize Vertex Endpoint
        ########################################################################

        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": api_endpoint}
        
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = vertex_ai.gapic.PredictionServiceClient(client_options=client_options)
        
        # The format of each instance should conform to the deployed model's prediction input schema.
        instances = instances if type(instances) == list else [instances]
        instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        
        endpoint = client.endpoint_path(
            project=project, location=location, endpoint=endpoint_id
        )
        
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        logging.info(f'Response: {response}')
        
        logging.info(f'Deployed Model ID(s): {response.deployed_model_id}')

        # The predictions are a google.protobuf.Value representation of the model's predictions.
        _predictions = response.predictions
        logging.info(f'Response Predictions: {_predictions}')
        
        return _predictions
    
    ################################################################################
    # Request Prediction
    ################################################################################

    TEST_INSTANCE_15 = {
        'album_name_can': 'Capoeira Electronica',
        'album_name_pl': [
            'Odilara', 'Capoeira Electronica', 'Capoeira Ultimate','Festa Popular', 'Capoeira Electronica',
            'Odilara', 'Capoeira Electronica', 'Capoeira Ultimate','Festa Popular', 'Capoeira Electronica',
            'Odilara', 'Capoeira Electronica', 'Capoeira Ultimate','Festa Popular', 'Capoeira Electronica'
        ],
        'album_uri_can': 'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR',
        'album_uri_pl': [
            'spotify:album:4Y8RfvZzCiApBCIZswj9Ry',
            'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR',
            'spotify:album:55HHBqZ2SefPeaENOgWxYK',
            'spotify:album:150L1V6UUT7fGUI3PbxpkE',
            'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR',
            'spotify:album:4Y8RfvZzCiApBCIZswj9Ry',
            'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR',
            'spotify:album:55HHBqZ2SefPeaENOgWxYK',
            'spotify:album:150L1V6UUT7fGUI3PbxpkE',
            'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR',
            'spotify:album:4Y8RfvZzCiApBCIZswj9Ry',
            'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR',
            'spotify:album:55HHBqZ2SefPeaENOgWxYK',
            'spotify:album:150L1V6UUT7fGUI3PbxpkE',
            'spotify:album:2FsSSHGt8JM0JgRy6ZX3kR'
        ],
        'artist_followers_can': 5170.0,
        'artist_genres_can': 'capoeira',
        'artist_genres_pl': [
            'samba moderno', 'capoeira', 'capoeira', 'NONE','capoeira',
            'samba moderno', 'capoeira', 'capoeira', 'NONE','capoeira',
            'samba moderno', 'capoeira', 'capoeira', 'NONE','capoeira'
        ],
        'artist_name_can': 'Capoeira Experience',
        'artist_name_pl': [
            'Odilara', 'Capoeira Experience', 'Denis Porto', 'Zambe','Capoeira Experience',
            'Odilara', 'Capoeira Experience', 'Denis Porto', 'Zambe','Capoeira Experience',
            'Odilara', 'Capoeira Experience', 'Denis Porto', 'Zambe','Capoeira Experience'
        ],
        'artist_pop_can': 24.0,
        'artist_pop_pl':[
            4., 24.,  2.,  0., 24.,
            4., 24.,  2.,  0., 24.,
            4., 24.,  2.,  0., 24.
        ],
        'artist_uri_can': 'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP',
        'artist_uri_pl': [
            'spotify:artist:72oameojLOPWYB7nB8rl6c',
            'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP',
            'spotify:artist:67p5GMYQZOgaAfx1YyttQk',
            'spotify:artist:4fH3OXCRcPsaHFE5KhgqZS',
            'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP',
            'spotify:artist:72oameojLOPWYB7nB8rl6c',
            'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP',
            'spotify:artist:67p5GMYQZOgaAfx1YyttQk',
            'spotify:artist:4fH3OXCRcPsaHFE5KhgqZS',
            'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP',
            'spotify:artist:72oameojLOPWYB7nB8rl6c',
            'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP',
            'spotify:artist:67p5GMYQZOgaAfx1YyttQk',
            'spotify:artist:4fH3OXCRcPsaHFE5KhgqZS',
            'spotify:artist:5SKEXbgzIdRl3gQJ23CnUP'
        ],
        'artists_followers_pl': [ 
            316., 5170.,  448.,   19., 5170.,
            316., 5170.,  448.,   19., 5170.,
            316., 5170.,  448.,   19., 5170.
        ],
        'duration_ms_can': 192640.0,
        'duration_ms_songs_pl': [234612., 226826., 203480., 287946., 271920., 234612., 226826., 203480., 287946., 271920., 234612., 226826., 203480., 287946., 271920.],
        'num_pl_albums_new': 9.0,
        'num_pl_artists_new': 5.0,
        'num_pl_songs_new': 85.0,
        'pl_collaborative_src': 'false',
        'pl_duration_ms_new': 17971314.0,
        'pl_name_src': 'Capoeira',
        'time_signature_can': '4',
        'time_signature_pl': ['4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4', '4'],
        'track_acousticness_can': 0.478,
        'track_acousticness_pl': [0.238 , 0.105 , 0.0242, 0.125 , 0.304, 0.238 , 0.105 , 0.0242, 0.125 , 0.304, 0.238 , 0.105 , 0.0242, 0.125 , 0.304 ],
        'track_danceability_can': 0.709,
        'track_danceability_pl': [0.703, 0.712, 0.806, 0.529, 0.821, 0.238 , 0.105 , 0.0242, 0.125 , 0.304, 0.238 , 0.105 , 0.0242, 0.125 , 0.304],
        'track_energy_can': 0.742,
        'track_energy_pl': [0.743, 0.41 , 0.794, 0.776, 0.947, 0.238 , 0.105 , 0.0242, 0.125 , 0.304, 0.238 , 0.105 , 0.0242, 0.125 , 0.304],
        'track_instrumentalness_can': 0.00297,
        'track_instrumentalness_pl': [4.84e-06, 4.30e-01, 7.42e-04, 4.01e-01, 5.07e-03, 4.84e-06, 4.30e-01, 7.42e-04, 4.01e-01, 5.07e-03, 4.84e-06, 4.30e-01, 7.42e-04, 4.01e-01, 5.07e-03],
        'track_key_can': '0',
        'track_key_pl': ['5', '0', '1', '10', '10', '5', '0', '1', '10', '10', '5', '0', '1', '10', '10'],
        'track_liveness_can': 0.0346,
        'track_liveness_pl': [0.128 , 0.0725, 0.191 , 0.105 , 0.0552,0.128 , 0.0725, 0.191 , 0.105 , 0.0552, 0.128 , 0.0725, 0.191 , 0.105 , 0.0552],
        'track_loudness_can': -7.295,
        'track_loudness_pl': [-8.638, -8.754, -9.084, -7.04 , -6.694, -8.638, -8.754, -9.084, -7.04 , -6.694, -8.638, -8.754, -9.084, -7.04 , -6.694],
        'track_mode_can': '1',
        'track_mode_pl': ['0', '1', '1', '0', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1'],
        'track_name_can': 'Bezouro Preto - Studio',
        'track_name_pl': [
            'O Telefone Tocou Novamente', 'Bem Devagar - Studio','Angola Dream', 'Janaina', 'Louco Berimbau - Studio',
            'O Telefone Tocou Novamente', 'Bem Devagar - Studio','Angola Dream', 'Janaina', 'Louco Berimbau - Studio',
            'O Telefone Tocou Novamente', 'Bem Devagar - Studio','Angola Dream', 'Janaina', 'Louco Berimbau - Studio'
        ],
        'track_pop_can': 3.0,
        'track_pop_pl': [5., 1., 0., 0., 1., 5., 1., 0., 0., 1., 5., 1., 0., 0., 1.],
        'track_speechiness_can': 0.0802,
        'track_speechiness_pl':[0.0367, 0.0272, 0.0407, 0.132 , 0.0734, 0.0367, 0.0272, 0.0407, 0.132 , 0.0734, 0.0367, 0.0272, 0.0407, 0.132 , 0.0734],
        'track_tempo_can': 172.238,
        'track_tempo_pl': [100.039,  89.089, 123.999, 119.963, 119.214, 100.039,  89.089, 123.999, 119.963, 119.214, 100.039,  89.089, 123.999, 119.963, 119.214],
        'track_uri_can': 'spotify:track:0tlhK4OvpHCYpReTABvKFb',
        'track_uri_pl': [
            'spotify:track:1pQkOdcTDfLr84TDCrmGy7',
            'spotify:track:39grEDsAHAjmo2QFo4G8D9',
            'spotify:track:5vxSLdJXqbKYH487YO8LSL',
            'spotify:track:6T9GbmZ6voDM4aTBsG5VDh',
            'spotify:track:7ELt9eslVvWo276pX2garN',
            'spotify:track:1pQkOdcTDfLr84TDCrmGy7',
            'spotify:track:39grEDsAHAjmo2QFo4G8D9',
            'spotify:track:5vxSLdJXqbKYH487YO8LSL',
            'spotify:track:6T9GbmZ6voDM4aTBsG5VDh',
            'spotify:track:7ELt9eslVvWo276pX2garN',
            'spotify:track:1pQkOdcTDfLr84TDCrmGy7',
            'spotify:track:39grEDsAHAjmo2QFo4G8D9',
            'spotify:track:5vxSLdJXqbKYH487YO8LSL',
            'spotify:track:6T9GbmZ6voDM4aTBsG5VDh',
            'spotify:track:7ELt9eslVvWo276pX2garN'
        ],
        'track_valence_can': 0.844,
        'track_valence_pl': [
            0.966, 0.667, 0.696, 0.876, 0.655,
            0.966, 0.667, 0.696, 0.876, 0.655,
            0.966, 0.667, 0.696, 0.876, 0.655
        ],
    }

    prediction_test = predict_custom_trained_model_sample(
        project=project,                     
        endpoint_id=endpoint_resource_path,
        location="us-central1",
        instances=TEST_INSTANCE_15
    )
    logging.info(f"prediction_test: {prediction_test}")
    
    ################################################################################
    # Init deployed indexes
    ################################################################################
    
    logging.info(f"ann_index_resource_uri: {ann_index_resource_uri}")
    logging.info(f"brute_force_index_resource_uri: {brute_force_index_resource_uri}")

    tree_ah_index = vertex_ai.MatchingEngineIndexEndpoint(index_name=ann_index_resource_uri)
    brute_force_index = vertex_ai.MatchingEngineIndexEndpoint(index_name=brute_force_index_resource_uri)
    
    DEPLOYED_ANN_INDEX_ID = tree_ah_index.deployed_indexes[0]
    DEPLOYED_BF_INDEX_ID = brute_force_index.deployed_indexes[0]
    
    logging.info(f"DEPLOYED_ANN_INDEX_ID: {DEPLOYED_ANN_INDEX_ID}")
    logging.info(f"DEPLOYED_BF_INDEX_ID: {DEPLOYED_BF_INDEX_ID}")
    
    ANN_response = deployed_ann_index.match(
        deployed_index_id=DEPLOYED_ANN_INDEX_ID,
        queries=prediction_test.predictions,
        num_neighbors=10
    )
    
    BF_response = deployed_bf_index.match(
        deployed_index_id=DEPLOYED_BF_INDEX_ID,
        queries=prediction_test.predictions,
        num_neighbors=10
    )
    
    # Calculate recall by determining how many neighbors were correctly retrieved as compared to the brute-force option.
    recalled_neighbors = 0
    for tree_ah_neighbors, brute_force_neighbors in zip(
        ANN_response, BF_response
    ):
        tree_ah_neighbor_ids = [neighbor.id for neighbor in tree_ah_neighbors]
        brute_force_neighbor_ids = [neighbor.id for neighbor in brute_force_neighbors]

        recalled_neighbors += len(
            set(tree_ah_neighbor_ids).intersection(brute_force_neighbor_ids)
        )

    recall = recalled_neighbors / len(
        [neighbor for neighbors in BF_response for neighbor in neighbors]
    )

    logging.info("Recall: {}".format(recall))
    
    metrics.log_metric("Recall", (recall * 100.0))
