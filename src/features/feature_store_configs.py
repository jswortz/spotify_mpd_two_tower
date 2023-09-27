TRACK_FEATURE_CONFIGS = {
    # "track_uri_can": {
    #     "value_type": "STRING",
    #     "description": "unique track ID; also uri",
    #     "labels": {"status": "passed"},
    # },
    "track_name_can": {
        "value_type": "STRING",
        "description": "name (text) of track",
        "labels": {"status": "passed"},
    },
    "artist_uri_can": {
        "value_type": "STRING",
        "description": "track's artist uri",
        "labels": {"status": "passed"},
    },
    "artist_name_can": {
        "value_type": "STRING",
        "description": "track's artist name (text)",
        "labels": {"status": "passed"},
    },
    "album_uri_can": {
        "value_type": "STRING",
        "description": "track's album uri",
        "labels": {"status": "passed"},
    },
    "album_name_can": {
        "value_type": "STRING",
        "description": "track's album name (text)",
        "labels": {"status": "passed"},
    },
    "duration_ms_can": {
        "value_type": "DOUBLE",
        "description": "track's duration in ms",
        "labels": {"status": "passed"},
    },
    "track_pop_can": {
        "value_type": "DOUBLE",
        "description": "track's popularity",
        "labels": {"status": "passed"},
    },
    "artist_pop_can": {
        "value_type": "DOUBLE",
        "description": "popularity of track's artist",
        "labels": {"status": "passed"},
    },
    "artist_genres_can": {
        "value_type": "STRING",  # STRING | STRING_ARRAY
        "description": "list of the genres associated with the track's artist",
        "labels": {"status": "passed"},
    },
    "artist_followers_can": {
        "value_type": "DOUBLE",
        "description": "number of followers for the track's artist",
        "labels": {"status": "passed"},
    },
    "track_danceability_can": {
        "value_type": "DOUBLE",
        "description": "describes how suitable a track is for dancing",
        "labels": {"status": "passed"},
    },
    "track_energy_can": {
        "value_type": "DOUBLE",
        "description": "represents the perceptual measure of intensity and activity",
        "labels": {"status": "passed"},
    },
    "track_key_can": {
        "value_type": "STRING",
        "description": "the key the track is in. Integers map to pitches using standard pitch class notation",
        "labels": {"status": "passed"},
    },
    "track_loudness_can": {
        "value_type": "DOUBLE",
        "description": "overall loudness of track in decibels; averaged across entire track",
        "labels": {"status": "passed"},
    },
    "track_mode_can": {
        "value_type": "STRING",
        "description": "the modality (major or minor) of a track (Major = 1; minor = 0)",
        "labels": {"status": "passed"},
    },
    "track_speechiness_can": {
        "value_type": "DOUBLE",
        "description": "detects the presence of spoken words ",
        "labels": {"status": "passed"},
    },
    "track_acousticness_can": {
        "value_type": "DOUBLE",
        "description": "a confidence measure from 0.0 to 1.0 whether the track is acoustic",
        "labels": {"status": "passed"},
    },
    "track_instrumentalness_can": {
        "value_type": "DOUBLE",
        "description": "predicts whether a track contains vocals",
        "labels": {"status": "passed"},
    },
    "track_liveness_can": {
        "value_type": "DOUBLE",
        "description": "detects the presence of an audience in the recording",
        "labels": {"status": "passed"},
    },
    "track_valence_can": {
        "value_type": "DOUBLE",
        "description": "measure describing the musical positiveness in a track",
        "labels": {"status": "passed"},
    },
    "track_tempo_can": {
        "value_type": "DOUBLE",
        "description": "the overall estimated tempo of a track in beats per minute (BPM)",
        "labels": {"status": "passed"},
    },
    "track_time_signature_can": {
        "value_type": "STRING",
        "description": "an estimated time signature (meter) describing how many beats per bar (e.g., 3/4 - 7/4)",
        "labels": {"status": "passed"},
    },
}

PLAYLIST_FEATURE_CONFIGS = {
    "pl_name_src": {
        "value_type": "STRING",
        "description": "name (text) of playlist",
        "labels": {"status": "passed"},
    },
    "pl_collaborative_src": {
        "value_type": "STRING",
        "description": "defines if multiple users can contribute to playlist",
        "labels": {"status": "passed"},
    },
    "pl_duration_ms_new": {
        "value_type": "DOUBLE",
        "description": "total duration of all the tracks in the playlist (in milliseconds)",
        "labels": {"status": "passed"},
    },
    "num_pl_songs_new": {
        "value_type": "DOUBLE",
        "description": "total duration of all the tracks in the playlist (in milliseconds)",
        "labels": {"status": "passed"},
    },
    "num_pl_artists_new": {
        "value_type": "DOUBLE",
        "description": "total duration of all the tracks in the playlist (in milliseconds)",
        "labels": {"status": "passed"},
    },
    "num_pl_albums_new": {
        "value_type": "DOUBLE",
        "description": "total duration of all the tracks in the playlist (in milliseconds)",
        "labels": {"status": "passed"},
    },
    # arrays
    "duration_ms_songs_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "track's duration in ms, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_pop_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "track's popularity, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "artist_pop_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "popularity of track's artist, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "artist_genres_pl": {
        "value_type": "STRING_ARRAY",  # STRING | STRING_ARRAY
        "description": "list of the genres associated with the track's artist, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "artist_followers_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "number of followers for the track's artist, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_danceability_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "describes how suitable a track is for dancing, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_energy_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "represents the perceptual measure of intensity and activity, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_key_pl": {
        "value_type": "STRING_ARRAY",
        "description": "the key the track is in. Integers map to pitches using standard pitch class notation, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_loudness_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "overall loudness of track in decibels; averaged across entire track, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_mode_pl": {
        "value_type": "STRING_ARRAY",
        "description": "the modality (major or minor) of a track (Major = 1; minor = 0), for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_speechiness_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "detects the presence of spoken words, for the last N tracks ",
        "labels": {"status": "passed"},
    },
    "track_acousticness_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "a confidence measure from 0.0 to 1.0 whether the track is acoustic, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_instrumentalness_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "predicts whether a track contains vocals, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_liveness_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "detects the presence of an audience in the recording, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_valence_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "measure describing the musical positiveness in a track, for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_tempo_pl": {
        "value_type": "DOUBLE_ARRAY",
        "description": "the overall estimated tempo of a track in beats per minute (BPM), for the last N tracks",
        "labels": {"status": "passed"},
    },
    "track_time_signature_pl": {
        "value_type": "STRING_ARRAY",
        "description": "an estimated time signature (meter) describing how many beats per bar, for the last N tracks",
        "labels": {"status": "passed"},
    },
}