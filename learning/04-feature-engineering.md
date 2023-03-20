# Feature Engineering

As the factorization-based models offer a pure collaborative filtering approach, the advanced feature processing with NDR architectures allow us to extend this to also incorporate aspects of [content-based filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics). By including additional features describing playlists and tracks, we give NDR models the opportunity to learn semantic concepts about `<playlist, track>` pairs. The ability to include label features (i.e., features about candidate tracks) also means our trained candidate tower can compute an embedding vector for candidate tracks not observed during training (i.e., cold-start). Conceptually, we can think of such a new candidate track embedding compiling all the content-based and collaborative filtering information learned from candidate tracks with the same or similar feature values.

With this flexibility to add multi-modal features, we just need to process them to produce embedding vectors with the same dimensions so they can be concatenated and fed to subsequent deep and cross layers. This means if we use pre-trained embeddings as an input feature, we would pass these through to the concatenation layer 


![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/2tower-feature-processing-f12.jpg)
> Illustration of feature processing from input to concatenated output. Text features are generated via n-grams. Integer indexes of n-grams are passed to an embedding layer. Hashing produces unique integers up to 1,000,000; values passed to an embedding layer. If using pre-trained embeddings, these are passed through the tower without transformation and concatenated with the other embedding representations.

## `Hashing` vs `StringLookup()` layers
Hashing is generally recommended when fast performance is needed and is preferred over string lookups because it skips the need for a lookup table. Setting the proper bin size for the hashing layer is critical. When there are more unique values than hashing bins, values start getting placed into the same bins, and this can negatively impact our recommendations. This is commonly referred to as a hashing collision, and can be avoided when building the model by allocating enough bins for the unique values. See [turning categorical features into embeddings](https://www.tensorflow.org/recommenders/examples/featurization#turning_categorical_features_into_embeddings) for more details. 

## `TextVectorization()` layers
The key to text features is to understand if creating additional NLP features with the [TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) layer is helpful. If additional context derived from the text feature is minimal, it may not be worth the cost to model training. This layer needs to be adapted from the source dataset, meaning the layer requires a scan of the training data to create lookup dictionaries for the top N n-grams (set by max_tokens). 

## feature engineering decision tree

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/feat-engineering-decision-tree-f13.jpg)
> Decision tree to guide feature engineering strategy