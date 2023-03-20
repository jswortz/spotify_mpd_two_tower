# Discussion on problem framing 
In this example, we use [MPD](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) to construct a recommendation use case, playlist continuation, where candidate tracks are recommended for a given playlist (query). This dataset is publicly available and offers several benefits for this demonstration:

* Includes real relationships between entities (e.g., playlists, tracks, artists) which can be difficult to replicate
* Large enough to replicate scalability issues likely to occur in production
* Variety of feature representations and data types (e.g., playlist and track IDs, raw text, numerical, datetime); ability to enrich dataset with additional metadata from the [Spotify Web Developer API](https://developer.spotify.com/documentation/web-api/)
* Teams can analyze the impact of modeling decisions by listening to retrieved candidate tracks (e.g., generate recommendations for your own Spotify playlists)

## Training examples
Creating training examples for recommendation systems is a non-trivial task. Like any ML use case, training data should accurately represent the underlying problem we are trying to solve. Failure to do this can lead to poor model performance and unintended consequences for the user experience. One such lesson from the [Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf) paper highlights that relying heavily on features such as ‘click-through rate’ can result in recommending clickbait (i.e., videos users rarely complete), as compared to features like ‘watch time’ which better capture a user’s engagement. 

Training examples should represent a semantic match in the data. For playlist-continuation, we can think of a semantic match as pairing playlists (i.e., a set of tracks, metadata, etc.) with tracks similar enough to keep the user engaged with their listening session. How does the structure of our training examples influence this?

* Training data is sourced from positive `<query, candidate>` pairs
* During training, we forward propagate query and candidate features through their respective towers to produce the two vector representations, from which we compute the dot product representing their similarity 
* After training, and before serving, the candidate tower is called to predict (precompute) embeddings for all candidate items
* At serving time, the model processes features for a given playlist and produces a vector embedding
* The playlist’s vector embedding is used in a search to find the most similar vectors in the precomputed candidate index
* The placement of candidate and playlist vectors in the embedding space, and the distance between them, is defined by the semantic relationships reflected in the training examples

The last point is important. Because the quality of our embedding space dictates the success of our retrieval, the model creating this embedding space needs to learn from training examples that best illustrate the relationship between a given playlist and ‘similar’ tracks to retrieve. 

This notion of similarity being highly dependent on the choice of paired data highlights the importance of preparing features that describe semantic matches. A model trained on <playlist title, track title> pairs will orient candidate tracks differently than a model trained on <aggregated playlist audio features, track audio features> pairs. 

Conceptually, training examples consisting of `<playlist title, track title>` pairs would create an embedding space in which all tracks belonging to playlists of the same or similar titles (e.g., ‘beach vibes’ and ‘beach tunes’) would be closer together than tracks belonging to different playlist titles (e.g., ‘beach vibes’ vs ‘workout tunes’); and examples consisting of `<aggregated playlist audio features, track audio features>` pairs would create an embedding space in which all tracks belonging to playlists with similar audio profiles (e.g., ‘live recordings of instrumental jams’ and ‘high energy instrumentals’) would be closer together than tracks belonging to playlists with different audio profiles (e.g., ‘live recordings of instrumental jams’ vs ‘acoustic tracks with lots of lyrics’).

The intuition for these examples is when we structure the rich track-playlist features in a format that describes how tracks show up on certain playlists, we can feed this data to a two tower model that learns all of the niche relationships with parent playlist and child tracks. Modern deep retrieval systems often consider user profiles, historical engagements, and context. While we don’t have user and context data in this example, they can easily be added to the query tower.