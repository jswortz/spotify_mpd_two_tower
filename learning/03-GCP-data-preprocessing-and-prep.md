# Preparing data with Google Cloud

## Data preparation with BigQuery
Constructing pairs of `<playlist, track>` features requires us to work with arrays and multidimensional data, which we can easily do with BigQuery. The high-level steps are outlined below. To skip directly to the code, refer to the [data prep notebook](https://github.com/jswortz/spotify_mpd_two_tower/blob/master/01-bq-data-prep.ipynb).

## Importing from JSON
The MPD comes in a JSON file where each line represents a playlist. Each playlist entry includes metadata features about the playlist (e.g., name, number of unique artists, etc.) and a track list, which includes metadata features about each track (e.g., its position in the playlist, duration, and unique IDs for artists, tracks, and albums). See the [README.md](https://github.com/jswortz/spotify_mpd_two_tower#info-on-data) for a complete data dictionary

From a [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction) notebook, we can load the raw JSON files to a pandas dataframe and push it to a BigQuery table using pandas-gbq. Each row in the resulting table should correspond to a single playlist, and the table schema should look like this:

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/bq-schema-load-json-raw-f8.jpg)
> BigQuery table schema after loading the MPD raw JSON files

Each playlist’s tracks are nested as one large string that needs to be parsed and converted to a [STRUCT](https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#struct_type). We can use BigQuery’s native [JSON parsing](https://cloud.google.com/bigquery/docs/reference/standard-sql/json_functions) for this:

```sql
CREATE OR REPLACE TABLE
  `{PROJECT_ID}.{bq_dataset}.playlists_nested` AS (
  WITH
    json_parsed AS (
    SELECT
      * EXCEPT(tracks),
      JSON_EXTRACT_ARRAY(tracks) AS json_data
    FROM
      `{PROJECT_ID}.{bq_dataset}.playlists` )
  SELECT
    json_parsed.* EXCEPT(json_data),
    ARRAY(
    SELECT
      AS STRUCT JSON_EXTRACT_SCALAR(json_data, "$.pos") AS pos, 
      JSON_EXTRACT_SCALAR(json_data, "$.artist_name") AS artist_name, 
      JSON_EXTRACT_SCALAR(json_data, "$.track_uri") AS track_uri, 
      JSON_EXTRACT_SCALAR(json_data, "$.artist_uri") AS artist_uri, 
      JSON_EXTRACT_SCALAR(json_data, "$.track_name") AS track_name, 
      JSON_EXTRACT_SCALAR(json_data, "$.album_uri") AS album_uri, 
      JSON_EXTRACT_SCALAR(json_data, "$.duration_ms") AS duration_ms, 
      JSON_EXTRACT_SCALAR(json_data, "$.album_name") AS album_name
    FROM
      json_parsed.json_data ) AS tracks,
  FROM
    json_parsed)
```

## Nested and repeated fields
After parsing the string data, each playlist entry has a `STRUCT` field named “tracks” that contains the complete track list and each track’s metadata. Data can now be accessed inside of the `STRUCT` via `UNNEST(tracks)`. See [Work with arrays](https://cloud.google.com/bigquery/docs/reference/standard-sql/arrays) to learn the advantages of using `ARRAYs` with `UNNEST`.

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/mpd-loaded-tracks-struct--f9.jpg)
> MPD loaded to BigQuery table, where “tracks” are formatted as a STRUCT

## Cross joining and selecting `<playlist, track>` pairs

We can create `<playlist, track>` pairs for playlist continuation by removing a track from the playlist’s `STRUCT` and placing it in a new column as the ‘candidate track’. 

* From a **user perspective**, we are trying to find good examples to continue the playlist (session) 
* From a **modeling perspective**, we can think of this candidate track as the example’s ‘label’ (i.e., what we are trying to predict/recommend given this playlist)
* If we translate this concept to the **embedding space**, we want the embedding vectors for the playlist and candidate track to be close together (i.e., given this playlist query, we should expect the candidate track among the nearest neighbors)

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/candidate-track-from-struct-f10.jpg)
> Create playlist’s candidate track by removing a track from the `STRUCT`

One approach is to create pairs for all children tracks and their parent playlists. For a single playlist, this means an example will be created such that each track in the playlist’s `STRUCT` will be considered the candidate track once. Assuming we apply a minimum playlist length of 5, an original playlist of 10 tracks would produce 5 `<playlist, track>` pairs.

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/pl-fan-out-f11.jpg)
> Assuming a minimum playlist length of 5 tracks, a 10-track playlist produces 5 `<playlist, track>` pairs

One drawback to this approach is that longer playlists will be oversampled compared to shorter playlists. As an example, consider a 10-track playlist and a 100-track playlist, both with the same playlist title “workout tunes”. If the 10-track playlist is composed of “pop” songs and the 100-track playlist is composed of “hair metal” songs, assuming all else equal, recommendations for playlists titled “workout tunes” will lean towards “hair metal” vs “pop”. To reduce this effect, we might sample only 5 `<playlists, track>` pairs for each original playlist ID.

We’ve included code for different sampling strategies and highly recommend evaluating candidate retrieval across different strategies. It’s also recommended to explore negative sampling strategies, which we have left out of scope for this example. 

## Data processing with Dataflow
Converting training data to TFRecords can help us achieve better input pipeline performance during model training. Once the train, validation, and candidate sets are created, we can use Dataflow to export these tables from BigQuery, serialize the examples to TFRecord format, and write them to a [Cloud Storage](https://cloud.google.com/storage) bucket. 

Note: Google provides a set of [open source](https://github.com/GoogleCloudPlatform/DataflowTemplates) Dataflow templates, including a [BigQuery to Cloud Storage TFRecords](https://cloud.google.com/dataflow/docs/guides/templates/provided-batch#bigquery-to-cloud-storage-tfrecords) batch template. However, at the time of writing, this template doesn’t support sequential data (e.g., playlist STRUCTS). Refer to the [tfrecord-beam-pipeline](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/02-tfrecord-beam-pipeline.ipynb) notebook to use a pipeline customized for this dataset.