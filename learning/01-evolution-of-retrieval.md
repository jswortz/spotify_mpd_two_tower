# Evolution of retrieval modeling

To better understand the benefits of two-tower architectures, let’s review three key modeling milestones in candidate retrieval.

## token-based matching
Traditional information retrieval systems rely heavily on token-based matching, where candidates are retrieved using an inverted index of n-grams. These systems are interpretable, easy to maintain (e.g., no training data), and are capable of achieving high precision. However, they typically suffer poor recall (i.e., trouble finding all relevant candidates for a given query) because they look for candidates having exact matches of key words. While they are still used for select Search use cases, many retrieval tasks today are either adapted with or replaced by embedding-based techniques.

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/token-based-retrieval.jpg)

> Token-based matching selects candidate items by matching key words found in both query and candidate items

## Factorization-based retrieval
Factorization-based retrieval introduces a simple embedding-based model that offers much better [generalization](https://developers.google.com/machine-learning/crash-course/generalization/video-lecture) by capturing the similarity between <query, candidate> pairs and mapping them to a shared embedding space. One of the major benefits to this [collaborative filtering](https://developers.google.com/machine-learning/recommendation/collaborative/basics) technique is that embeddings are learned automatically from implicit query-candidate interactions. Fundamentally, these models factorize the full query-candidate interaction (co-occurrence) matrix to produce smaller, dense embedding representations of queries and candidates, where the product of these embedding vectors is a good approximation of the interaction matrix. The idea is that by compacting the full matrix into k dimensions the model learns the top k latent factors describing `<query, candidate>` pairs with respect to the modeling task.

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/factorization-based-retrieval.jpg)

> Factorization-based models factorize a query-candidate interaction matrix into the product of two lower-rank matrices that capture the query-candidate interactions

## Neural Deep Retrieval (NDR)
The latest modeling paradigm for retrieval, commonly referred to as neural deep retrieval (NDR), produces the same embedding representations, but uses deep learning to create them. NDR models like two-tower encoders apply deep learning by processing input features with successive network layers to learn layered representations of the data. Effectively, this results in a neural network that acts as an information distillation pipeline, where raw, multi-modal features are repeatedly transformed such that useful information is magnified and irrelevant information is filtered. This results in a highly expressive model capable of learning non-linear relationships and more complex feature interactions. These options essentially generalize the collaborative filtering approach

![alt text](https://github.com/jswortz/spotify_mpd_two_tower/blob/main/img/ndr-retrieval.jpg)

> NDR architectures like two-tower encoders are conceptually similar to factorization models. Both are embedding-based retrieval techniques computing lower-dimensional vector representations of query and candidates, where the similarity between these two vectors is determined by computing their dot product.

### Two-tower retrieval
In a two-tower architecture, each tower is a neural network that processes either query or candidate input features to produce an embedding representation of those features. Because the embedding representations are simply vectors of the same length, we can compute the [dot product](https://developers.google.com/machine-learning/recommendation/overview/candidate-generation#dot-product) between these two vectors to determine how close they are. This means the orientation of the embedding space is determined by the dot product of each `<query, candidate>` pair in the training examples.