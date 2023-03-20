# Feature Engineering

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/2tower-feature-processing-f12.jpg)

## `Hashing` vs `StringLookup()` layers
Hashing is generally recommended when fast performance is needed and is preferred over string lookups because it skips the need for a lookup table. Setting the proper bin size for the hashing layer is critical. When there are more unique values than hashing bins, values start getting placed into the same bins, and this can negatively impact our recommendations. This is commonly referred to as a hashing collision, and can be avoided when building the model by allocating enough bins for the unique values. See [turning categorical features into embeddings](https://www.tensorflow.org/recommenders/examples/featurization#turning_categorical_features_into_embeddings) for more details. 

## `TextVectorization()` layers
The key to text features is to understand if creating additional NLP features with the [TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) layer is helpful. If additional context derived from the text feature is minimal, it may not be worth the cost to model training. This layer needs to be adapted from the source dataset, meaning the layer requires a scan of the training data to create lookup dictionaries for the top N n-grams (set by max_tokens). 

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/feat-engineering-decision-tree-f13.jpg)
> Decision tree to guide feature engineering strategy