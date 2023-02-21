import tensorflow as tf

candidate_features = {
    "track_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),            
    "track_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "album_uri_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),           
    "album_name_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()), 
    "duration_ms_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
    "track_pop_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),      
    "artist_pop_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "artist_genres_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "artist_followers_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    # new
    # "track_pl_titles_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_danceability_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_energy_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_key_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_loudness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_mode_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
    "track_speechiness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_acousticness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_instrumentalness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_liveness_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_valence_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "track_tempo_can":tf.io.FixedLenFeature(dtype=tf.float32, shape=()),
    "time_signature_can":tf.io.FixedLenFeature(dtype=tf.string, shape=()),
}