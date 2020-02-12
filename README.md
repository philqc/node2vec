# node2vec: Scalable Feature Learning for Networks
This is a Python implementation of the paper [node2vec: Scalable Feature Learning for Networks](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf) accepted in KDD2016.

## Example Tasks 

### (1) Feature extraction with Relations dataset
In a Data Science class (IFT-6758 at UdeM/Mila), we were given the task to predict age, gender 
and personality traits of users from a social network website. One of the data sources was a Relation.csv
file, which was essentially an edge list between users and liked pages. We implemented Node2Vec
to do feature extraction on the relational data, which was subsequently concatenated with other data
sources (pictures, text) in a multimodal approach. Since the data is private, we incorporated two fake 
Relations.csv in ``` .tests/Relation/``` that show how the data was constructed. To run the model,
simply write
```bash
python -m src.learn_features --type relation
```
Other hyparameters can be specified (min_like, p, q values of node2vec biased random walk) 
as command-line arguments.
### (2) Multi-Label Classfication with BlogCatalog dataset
We also reproduced the results from node2vec paper on the BlogCatalog dataset to test our implementation.
To run the feature extraction run the command:
```bash
python -m src.learn_features --type blogcatalog --p 0.25 --q 0. --min_like 0
```


## Authors
* **PHILLIPPE BEARDSELL** - *Professional Machine Learning Master Student at [Mila](https://mila.quebec/)* 
* **HSU, CHIH-CHAO** - *Professional Machine Learning Master Student at [Mila](https://mila.quebec/)* 

## Acknowlegements

## To Do
### Try multiprocessing for neigbor prob calculation
### Implement Alias Sampling
### Evaluation code (predic_prob)
