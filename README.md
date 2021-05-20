## GraphReach : Position-Aware Graph Neural Network using Reachability Estimations

There are two code folders one each for GraphReach and P-GNN with adversarial attack functionality.

The P-GNN framework was taken from [https://github.com/JiaxuanYou/P-GNN] and has other
benchmarks implemented as well. The framework code is added as
the adversarial module is additional to the existing prior work.


## Experimental Setup
Python 3.7.6
Pytorch 1.4.0
NetworkX 2.3
Cuda 10.0


## Libraries Required
Pytorch
PyTorch Geometric
torch-scatter
torch-sparse
torch-cluster
torch-spline-conv (optional)
networkx
tensorboardx

## USAGE
The models can be run using the shell file in respective folders.
USAGE: ./cmds.sh


## Arguments:

Note: Arguments related to random walks correspond to GraphReach only

--model
The model to be executed
Options: GraphReach, PGNN, GCN, MLP, SAGE, GAT, GIN

--dataset
Dataset to train the model
Options: grid, communities, ppi, email, completeEmail, protein, All

--task
Task to perform, link prediction (LP) or pairwise node classification (PNC)
Options: link (for LP), link_pair (for PNC)

--remove_link_ratio
Fraction of data to be retained as validation and test set
(Validation and test are split into half)

--layer_num
Number of layers in model

--epoch_num 
Number of epochs

--repeat_num
Number of iterations of experiment

--num_walk
Number of random walks

--walk_length
Length of the random walk 

--unweightedRandomWalk
For unweighted random walk

--weightedRandomWalk
For weighted random walk

--edgelabel
To bias random walk
Used only for pairwise node classification task
Not to be used in adversarial setting

--Num_Anchors
Number of anchors
Options: 3, 5, 8, logn, 2logn, 5logn, logn2
(logn2 for (log(n))^2 anchors) 

--select_anchors
Anchor Selection Strategy
Options: DiversifiedRandomK, DiversifiedTopK, topK, random
(For GraphReach only)

--sampleXwalks
Sample X% of walks from all the random walks
(For diversified anchor selection)

--sampleMbigraphs
Sample M Bipartite Graphs
(For diversified anchor selection)

--rm_feature
For inductive setting, default is tranductive

--attention
For attention based aggregation
(GraphReach only)

--attentionAddSelf
For attention based aggregation with self feature
(GraphReach only)

--AdversarialAttack
Perform adversarial testing

--addFedges
Add false edges to nodes involved in sampled fraction of test pairs


## Testing on Unseen Graph
The model can be run on completely new unseen graph by uncommenting Line#51 in test.py.
USAGE: ./test.sh


## Data:

Data used for the experiments can be downloaded from the following link.
https://github.com/JiaxuanYou/P-GNN/tree/master/data
