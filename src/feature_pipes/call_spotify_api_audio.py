
import kfp
from typing import Any, Callable, Dict, NamedTuple, Optional, List
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, component, Metrics)

@kfp.v2.dsl.component(
    base_image="python:3.9",
    packages_to_install=[
        'fsspec', 'google-cloud-bigquery',
        'google-cloud-storage',
        'gcsfs',
        'spotipy','requests','db-dtypes',
        'numpy','pandas','pyarrow','absl-py', 'pandas-gbq==0.17.4',
        'tqdm'
    ]
)
def call_spotify_api_audio(
    project: str,
    location: str,
    client_id: str,
    batch_size: int,
    batches_to_store: int,
    target_table: str,
    client_secret: str,
    unique_table: str,
    sleep_param: float,
) -> NamedTuple('Outputs', [
    ('done_message', str),
]):
    print(f'pip install complete')
    import os
    
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    
    import re
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    import pandas as pd
    import json
    import time
    import numpy as np
    
    from requests.exceptions import ReadTimeout, HTTPError, ConnectionError, RequestException
    from absl import logging
    
    from google.cloud import storage
    import gcsfs
    from google.cloud import bigquery
    
    import pandas_gbq
    from multiprocessing import Process
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    
    from google.cloud.exceptions import NotFound

    import multiprocessing

    # print(f'package import complete')

    logging.set_verbosity(logging.INFO)
    logging.info(f'package import complete')

    
    bq_client = bigquery.Client(
      project=project, location=location
    )
    
    def spot_audio_features(uri, client_id, client_secret):

        # Authenticate
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id, 
            client_secret=client_secret
        )
        sp = spotipy.Spotify(
            client_credentials_manager = client_credentials_manager, 
            requests_timeout=10, 
            retries=10
        )
        ############################################################################
        # Create Track Audio Features DF
        ############################################################################
        
        uri_stripped = [u.replace('spotify:track:', '') for u in uri] #fix the quotes 
        #getting track popularity
        tracks = sp.tracks(uri_stripped)
        #Audio features
        time.sleep(sleep_param)
    
        a_feats = sp.audio_features(uri)
        features = pd.json_normalize(a_feats)#.to_dict('list')
        
        features['track_pop'] = pd.json_normalize(tracks['tracks'])['popularity']
        
        features['track_uri'] = uri
        return features

    bq_client = bigquery.Client(
        project=project, 
        location='US'
    )
    
    #check if target table exists and if so return a list to not duplicate records
    try:
        bq_client.get_table(target_table)  # Make an API request.
        logging.info("Table {} already exists.".format(target_table))
        target_table_incomplete_query = f"select distinct track_uri from `{target_table}`"
        loaded_tracks_df = bq_client.query(target_table_incomplete_query).result().to_dataframe()
        loaded_tracks = loaded_tracks_df.track_uri.to_list()
        
    except NotFound:
        logging.info("Table {} is not found.".format(target_table))
    
    query = f"select distinct track_uri from `{unique_table}`" 

    #refactor
    schema = [{'name':'danceability', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'energy', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'key', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'loudness', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'mode', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name':'speechiness', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'acousticness', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'instrumentalness', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'liveness', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'valence', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'tempo', 'type': 'FLOAT', "mode": "NULLABLE"},
        {'name':'type', 'type': 'STRING', "mode": "NULLABLE"},
        {'name':'id', 'type': 'STRING', "mode": "NULLABLE"},
        {'name':'uri', 'type': 'STRING', "mode": "NULLABLE"},
        {'name':'track_href', 'type': 'STRING', "mode": "NULLABLE"},
        {'name':'analysis_url', 'type': 'STRING', "mode": "NULLABLE"},
        {'name':'duration_ms', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name':'time_signature', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name':'track_pop', 'type': 'INTEGER', "mode": "NULLABLE"},
        {'name':'track_uri', 'type': 'STRING', "mode": "REQUIRED"},
    ]
    
    tracks = bq_client.query(query).result().to_dataframe()
    track_list = tracks.track_uri.to_list()
    logging.info(f'finished downloading tracks')
    
    
    ### This section is used when there are tracks already loaded into BQ and you want to resume loading the data
    try:
        track_list = list(set(track_list) - set(loaded_tracks)) #sets the new track list to remove already loaded data in BQ
    except:
        pass
    

    from tqdm import tqdm
    def process_track_list(track_list):
        
        uri_list_length = len(track_list)-1 #starting count at zero
        inner_batch_count = 0 #avoiding calling the api on 0th iteration
        uri_batch = []
        
        for i, uri in enumerate(tqdm(track_list)):
            uri_batch.append(uri)
            if (len(uri_batch) == batch_size or uri_list_length == i) and i > 0: #grab a batch of 50 songs
                    # logging.info(f"appending final record for nth song at: {inner_batch_count} \n i: {i} \n uri_batch length: {len(uri_batch)}")
                    ### Try catch block for function
                try:
                    audio_featureDF = spot_audio_features(uri_batch, client_id, client_secret)
                    time.sleep(sleep_param)
                    uri_batch = []
                except ReadTimeout:
                    logging.info("'Spotify timed out... trying again...'")
                    audio_featureDF = spot_audio_features(uri_batch, client_id, client_secret)
                    
                    uri_batch = []
                    time.sleep(sleep_param)
                
                except HTTPError as err: #JW ADDED
                    logging.info(f"HTTP Error: {err}")
                
                except spotipy.exceptions.SpotifyException as spotify_error: #jw_added
                    logging.info(f"Spotify error: {spotify_error}")
                    
                # Accumulate batches on the machine before writing to BQ
                # if inner_batch_count <= batches_to_store or uri_list_length == i:
                if inner_batch_count == 0:
                    appended_data = audio_featureDF
                    # logging.info(f"creating new appended data at IBC: {inner_batch_count} \n i: {i}")
                    inner_batch_count += 1
                elif uri_list_length == i or inner_batch_count == batches_to_store: #send the batches to bq
                    appended_data = pd.concat([audio_featureDF, appended_data])
                    inner_batch_count = 0
                    appended_data.to_gbq(
                        destination_table=target_table, 
                        project_id=f'{project}', 
                        location='US', 
                        table_schema=schema,
                        progress_bar=False, 
                        reauth=False, 
                        if_exists='append'
                    )
                    logging.info(f'{i+1} of {uri_list_length} complete!')
                else:
                    appended_data = pd.concat([audio_featureDF, appended_data])
                    inner_batch_count += 1

        logging.info(f'audio features appended')
    
    #multiprocessing portion - we will loop based on the modulus of the track_uri list
    #chunk the list 
    
    # Yield successive n-sized
    # chunks from l.
    def divide_chunks(l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]
            
    n_cores = multiprocessing.cpu_count() 
    chunked_tracks = list(divide_chunks(track_list, int(len(track_list)/n_cores))) #produces a list of lists chunked evenly by groups of n_cores
    
    logging.info(
        f"""
        total tracks downloaded: {len(track_list)}\n
        length of chunked_tracks: {len(chunked_tracks)}\n 
        and inner dims: {[len(x) for x in chunked_tracks]}
        """
    )

    procs = []
    
    def create_job(target, *args):
        p = multiprocessing.Process(target=target, args=args)
        p.start()
        return p

    # starting process with arguments
    for track_chunk in chunked_tracks:
        proc = create_job(process_track_list, track_chunk)
        time.sleep(np.pi)
        procs.append(proc)

    # complete the processes
    for proc in procs:
        proc.join()
        
    # process_track_list(track_list) #single thread
     
    logging.info(f'audio features appended')
    
    return (
          f'DONE',
      )
