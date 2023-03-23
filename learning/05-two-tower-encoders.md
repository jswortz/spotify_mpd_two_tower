# Two-tower encoders for retrieval
In a two-tower architecture, each tower is a neural network that processes either query or candidate input features to produce an embedding representation of those features. Because the embedding representations are simply vectors of the same length, we can compute the [dot product](https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#dot-product) between these two vectors to determine how close they are. This means the orientation of the embedding space is determined by the dot product of each `<query, candidate>` pair in the training examples.

### TODO
* add some more