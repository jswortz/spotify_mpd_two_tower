## multi-stage recsys: ranking

```
A ranking problem is different from classification/regression in which the objective is to optimize for the correctness of the relative order for a list of examples (e.g., documents) given a context (e.g., a query), but not the absolute errors on individual examples. In addition, it is often more important to get the top-ranking items correct in a ranking problem compared to items at the bottom of a ranked list, while classification/regression problems often weigh every individual item equally.
```

### Listwise model [(tfrs docs)](https://www.tensorflow.org/recommenders/examples/listwise_ranking#listwise_model)
* The `ListMLE` loss from TensorFlow Ranking expresses list maximum likelihood estimation. 
* To calculate the ListMLE loss, we first use the user ratings to generate an optimal ranking. 
* We then calculate the likelihood of each candidate being out-ranked by any item below it in the optimal ranking using the predicted scores. 
* The model tries to minimize such likelihood to ensure highly rated candidates are not out-ranked by low rated candidates. 
* Note that since the likelihood is computed with respect to a candidate and all candidates below it in the optimal ranking, the loss is not pairwise but listwise. Hence the training uses list optimization.

> see details of ListMLE in section 2.2 of [Position-aware ListMLE: A Sequential Learning Process](http://auai.org/uai2014/proceedings/individuals/164.pdf)