import json
import tensorflow as tf
import tensorflow_recommenders as tfrs

from google.cloud import storage

import numpy as np
import pickle as pkl
from pprint import pprint

keep_feats = [
'name',
'collaborative',
"track_uri_can",
"n_songs_pl",
'num_artists_pl',
"num_albums_pl",
"artist_name_pl",
"track_uri_pl",
"track_name_pl",
"duration_ms_songs_pl",
"album_name_pl",
"artist_pop_pl",
"artists_followers_pl",
"track_pop_pl",
"artist_genres_pl",
'artist_name_can',
'track_name_can',
'album_name_can',
'artist_uri_can',
'track_uri_can',
'album_uri_can',
'duration_ms_can',
'track_pop_can',
'artist_pop_can',
'artist_followers_can',
'album_uri_can',
]

MAX_PLAYLIST_LENGTH = 375
BUCKET_NAME = 'spotify-v1'
FILE_PATH = 'vocabs/v2_string_vocabs'
FILE_NAME = 'string_vocabs_v1_20220924-tokens22.pkl'
DESTINATION_FILE = 'downloaded_vocabs.txt'
client = storage.Client()

with open(f'{DESTINATION_FILE}', 'wb') as file_obj:
    client.download_blob_to_file(
        f'gs://{BUCKET_NAME}/{FILE_PATH}/{FILE_NAME}', file_obj)

    
with open(f'{DESTINATION_FILE}', 'rb') as pickle_file:
    vocab_dict = pkl.load(pickle_file)
    
# query/playlist inputs
def pre_hash_records(data, candidate_mode=False):
    
    #candidate data stuff
    
    track_uri_can_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["track_uri_can"]))(data["track_uri_can"])
    artist_name_can_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["artist_name_can"]))(data['artist_name_can'])
    track_name_can_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["track_name_can"]))(data['track_name_can'])
    album_name_can_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["album_name_can"]), mask_value='')(data['album_name_can'])
    # album_uri_can 
    album_uri_can_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["album_uri_can"]), mask_value='')(data['album_uri_can'])
    # artist_uri_can
    artist_uri_can_hashed = tf.keras.layers.Hashing(num_bins=200_000)(data['artist_uri_can'])
    # artist_genres_can_hashed = tf.keras.layers.Hashing(num_bins=200_000)(data['artist_genres_can'])
    
    data["artist_name_can"] = artist_name_can_hashed
    data["track_uri_can"] = track_uri_can_hashed
    data["track_name_can"] = track_name_can_hashed
    data["album_name_can"] = album_name_can_hashed
    data["track_uri_can"] = track_uri_can_hashed
    data['artist_uri_can'] = artist_uri_can_hashed
    data['album_uri_can'] = album_uri_can_hashed
    # data["artist_genres_can"] = artist_genres_can
    if candidate_mode:
        return(data)
    else:
    #end candidate hashing
        artist_name_pl = tf.reshape(data["artist_name_pl"], (-1, MAX_PLAYLIST_LENGTH))
        track_uri_pl = tf.reshape(data["track_uri_pl"], (-1, MAX_PLAYLIST_LENGTH))
        track_name_pl = tf.reshape(data["track_name_pl"], (-1, MAX_PLAYLIST_LENGTH))
        album_name_pl = tf.reshape(data["album_name_pl"], (-1, MAX_PLAYLIST_LENGTH))
        artist_genres_pl = tf.reshape(data["artist_genres_pl"], (-1, MAX_PLAYLIST_LENGTH))

        # name
        name_hashed = tf.keras.layers.Hashing(num_bins=1_000_000)(data['name'])
        # collaborative
        collab_hashed = tf.keras.layers.Hashing(num_bins=3)(data['collaborative'])
        #track_uri_can and pl
        track_uri_pl_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["track_uri_can"]), mask_value='')(track_uri_pl)
        #artist_name_can and pl
        artist_name_pl_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["artist_name_can"]), mask_value='')(artist_name_pl)
        # track_name_can and pl
        track_name_pl_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["track_name_can"]), mask_value='')(track_name_pl)
        #album_name_can and pl
        album_name_pl_hashed = tf.keras.layers.Hashing(num_bins=len(vocab_dict["album_name_can"]), mask_value='')(album_name_pl)
        # artist_genres_can and pl
        artist_genres_pl_hashed = tf.keras.layers.Hashing(num_bins=200_000, mask_value='')(artist_genres_pl)

        #replace data with hashed values
        data['name'] = name_hashed
        data['collaborative'] = collab_hashed
        data["artist_name_pl"] = artist_name_pl_hashed
        data["track_uri_pl"] = track_uri_pl_hashed
        data["track_name_pl"] = track_name_pl_hashed
        data["album_name_pl"] = album_name_pl_hashed
        data["artist_genres_pl"] = artist_genres_pl_hashed
        new_data = {}
        for k in keep_feats:
            new_data[k] = data[k]
        return(new_data)
