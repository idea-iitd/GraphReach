import torch
import networkx as nx
import numpy as np
import multiprocessing as mp
import random
import node2vec
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms import bipartite
from numpy import *
import matplotlib.pyplot as plt 
from itertools import chain
from args import *
####################################################################
args1 = make_args()
# # approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negtive_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0],mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes,num_nodes)
    prob = num_negtive_edges / (num_nodes*num_nodes-mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp<prob))
    return mask_link_negative


def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    mask_link_positive_rev=[]
    mask_link_positive_rev=np.array([mask_link_positive[1],mask_link_positive[0]])

    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
        mask_link_positive_set.append(tuple(mask_link_positive_rev[:,i]))

    mask_link_positive_set = set(mask_link_positive_set)
    
    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break


    return mask_link_negative

def resample_edge_mask_link_negative(data):
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_train.shape[1])
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                     num_negtive_edges=data.mask_link_positive_test.shape[1])


def deduplicate_edges(edges):
    edges_new = np.zeros((2,edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = {} # node already put into result
    for i in range(edges.shape[1]):
        if edges[0,i]<edges[1,i]:
            edges_new[:,j] = edges[:,i]
            j += 1
        elif edges[0,i]==edges[1,i] and edges[0,i] not in skip_node:
            edges_new[:,j] = edges[:,i]
            skip_node.add(edges[0,i])
            j += 1

    return edges_new

def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1,:]), axis=-1)

def split_for_NodeClassification(nodes,remove_ratio):
    random_nodes = np.random.permutation(nodes)
    e = len(random_nodes)
    split1 = int((1-remove_ratio)*e)
    split2 = int((1-remove_ratio/2)*e)
    nodes_train = random_nodes[:split1]
    nodes_val = random_nodes[split1:split2]
    nodes_test = random_nodes[split2:]

    return nodes_train, nodes_val, nodes_test

# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test


def edge_to_set(edges):
    edge_set = []
    for i in range(edges.shape[1]):
        edge_set.append(tuple(edges[:, i]))
    edge_set = set(edge_set)
    return edge_set


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def add_nx_graph(data):
    G = nx.Graph()
    edge_numpy = data.edge_index.numpy()
    edge_list = []
    for i in range(data.num_edges):
        edge_list.append(tuple(edge_numpy[:, i]))
    G.add_edges_from(edge_list)
    data.G = G


def graph_random_walks(i,edge_labels,args,graph,val_data,test_data):

    if args.edgelabel:
        if edge_labels[0] is not None:

            for edge in graph.edges():
                if edge_labels[i][edge[0]][edge[1]]!=0:
                    graph[edge[0]][edge[1]]['weight'] = 1/edge_labels[i][edge[0]][edge[1]] 
                else:
                    graph[edge[0]][edge[1]]['weight']=0.01
        else:
            for edge in graph.edges():
                graph[edge[0]][edge[1]]['weight'] = 1
    else:
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = 1


    G = node2vec.Graph(graph, args.directed, args.p, args.q,args.fastRandomWalk)
    G.preprocess_transition_probs()
    theta=2.0
    st_node,walks = G.simulate_walks(args.num_walks, args.walk_length,theta)

    return st_node,walks

import random

def sampleFromLists(lists,n):
    """draws n elements from each list in lists
returning the result as a single list"""
    sample = []
    for subList in lists:
        sample.extend(random.sample(subList,n))
    return sample


#Reachability Computation Function
def precompute_dist_data(i,edge_labels,args,edge_index, num_nodes,val_data,test_data):
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)
        
        n = num_nodes
        dists_array = np.zeros((n, n))   

        bipartite_graph_RW_mat_list = []
        m_bigraphs = args.sampleMbigraphs
        
        st_node,walks = graph_random_walks(i,edge_labels,args,graph,val_data,test_data)
        
        dists_dict = {}
        start_node_count ={}
        for node in graph.nodes():
        	start_node_count[node]=0
        for node in st_node:
                start_node_count[node]= start_node_count[node]+1
        

        ########################################################################
        ####Sampling X number of  walks m times
        for idx in range(m_bigraphs):
            bipartite_graph_RW_mat= np.zeros((n, n))
            X=args.sampleXwalks
            total_walks=len(walks)
            sampled_randomwalk_count=int(X*total_walks)
        
            Sampled_X_random_walks= random.sample(walks,sampled_randomwalk_count)

            for index,Sampled_X_walk in enumerate(Sampled_X_random_walks):
                for wt,node_index in enumerate(Sampled_X_walk):
                        bipartite_graph_RW_mat[node_index][Sampled_X_walk[0]]+=1  #STORE COLUMN-WISE
                    
            bipartite_graph_RW_mat_list.append(bipartite_graph_RW_mat)

        ########################################################################
        
        
        for x,walk in enumerate(walks):
            for wt,node in enumerate(walk):
                if args.weightedRandomWalk:
                    dists_array[node][st_node[x]]+=1/(wt+1)
                else:
                    dists_array[node][st_node[x]]+=1

        if args.normalized:
            max_col=np.max(dists_array, axis=0)
            dists_array=dists_array/max_col
            where_are_NaNs = isnan(dists_array)
            dists_array[where_are_NaNs] = 0
        

        return dists_array,bipartite_graph_RW_mat_list


def get_Random__single_anchor_node(n,dist,device,c=0.5):
    if args1.Num_Anchors=='logn':
        m = int(np.log2(n))
        copy = c
    elif args1.Num_Anchors=='logn2':
        m = int(np.log2(n))
        copy = int(c*m)
    elif args1.Num_Anchors=='3':
        m = 3
        copy = c
    elif args1.Num_Anchors=='5':
        m = 5
        copy = c
    elif args1.Num_Anchors=='8':
        m = 8
        copy = c
    elif args1.Num_Anchors=='2logn':
        m = int(np.log2(n))
        copy = int(c*2)
    elif args1.Num_Anchors=='5logn':
        m = int(np.log2(n))
        copy = int(c*5)
    elif args1.Num_Anchors=='500':
        if n>=500:
            x = 500
            y = n//2
            if x>y:
                m=y
            else:
                m=x
        else:
            m=n//2
        copy = c
        

    anchorset_id = []
    no_of_anchors = m*copy
    if no_of_anchors>=n:
        no_of_anchors=n
        for anchor_ix in range(no_of_anchors):
            ix=np.array([anchor_ix])
            anchorset_id.append(ix)

    else:
        
        new_anchorset_id = []
        x=[]
        for i in range(m):
            anchor_size = 1        
            j=0
            while  j<copy:
                x=np.random.choice(n,size=anchor_size,replace=False)
                if x not in anchorset_id:
                    anchorset_id.append(x)
                    j+=1
                else:
                    x=np.random.choice(n,size=anchor_size,replace=False)

    return anchorset_id


def get_TopK__single_anchor_node(n,dist,device,c=0.5):
    if args1.Num_Anchors=='logn':
        m = int(np.log2(n))
        copy = c
    elif args1.Num_Anchors=='logn2':
        m = int(np.log2(n))
        copy = int(c*m)
    elif args1.Num_Anchors=='3':
        m = 3
        copy = c
    elif args1.Num_Anchors=='5':
        m = 5
        copy = c
    elif args1.Num_Anchors=='8':
        m = 8
        copy = c
    elif args1.Num_Anchors=='2logn':
        m = int(np.log2(n))
        copy = int(c*2)
    elif args1.Num_Anchors=='5logn':
        m = int(np.log2(n))
        copy = int(c*5)
    elif args1.Num_Anchors=='500':
        if n>=500:
            x = 500
            y = n//2
            if x>y:
                m=y
            else:
                m=x
        else:
            m=n//2
        copy = c
    anchorset_id = []
    new_anchorset_id = []
    no_of_anchors = m*copy
    if no_of_anchors>=n:
        no_of_anchors=n
        for anchor_ix in range(no_of_anchors):
            ix=np.array([anchor_ix])
            new_anchorset_id.append(ix)

    else:

        
        x=[]

        dist_max_temp = torch.zeros((dist.shape[0],dist.shape[1])).to(device)
        dist_max_temp = torch.sum(dist, dim=1)

        sorted_dist=torch.sort(dist_max_temp, dim=0, descending=True)
        sorted_anchor_nodes=sorted_dist.indices
        sorted_anchor=sorted_anchor_nodes.cpu()
        sorted_anchor = sorted_anchor.numpy()
        number_of_nodes=len(sorted_anchor)

        j=0
        x1=[]
        while j<(m*copy): 
                x1=np.array([sorted_anchor[j] ])
                new_anchorset_id.append(x1)
                j+=1

    return new_anchorset_id

 # Diversified anchor selection
def get_orthoTopK(bipartite_graph_RW_input_mat_list,top3K_anchorset_id,args,n,dist,device,c=1):        # Diversified anchor selection
    if args1.Num_Anchors=='logn':
        m = int(np.log2(n))
        copy = c#int(c*m)
    elif args1.Num_Anchors=='logn2':
        m = int(np.log2(n))
        copy = int(c*m)
    elif args1.Num_Anchors=='3':
        m = 3
        copy = c#int(c*m)
    elif args1.Num_Anchors=='5':
        m = 5
        copy = c#int(c*m)
    elif args1.Num_Anchors=='8':
        m = 8
        copy = c#int(c*m)
    elif args1.Num_Anchors=='2logn':
        m = int(np.log2(n))
        copy = int(c*2)#int(c*m)
    elif args1.Num_Anchors=='5logn':
        m = int(np.log2(n))
        copy = int(c*5)#int(c*m)
    elif args1.Num_Anchors=='500':
        if n>=500:
            x = 500
            y = n//2
            if x>y:
                m=y
            else:
                m=x
        else:
            m=n//2
        copy = c#int(c*m)

    no_of_anchors = m*copy
    if no_of_anchors>n:
        no_of_anchors=n
    new_anchorset_id = []
    cumulative_score = dict()

    for idx_biG, bigraph_mat in enumerate(bipartite_graph_RW_input_mat_list):
        edge_list=[]
        r,c = bigraph_mat.shape
        # adjacency list of bipartite grap stored in dict form
        # type(values) = list, keys = visited vertices, value = origin vertices
        bigraph_adjList = dict()
        for idx_c in range(c):
            for idx_r in range(r):
                if(bigraph_mat[idx_r][idx_c]!=0):
                    if(idx_r not in bigraph_adjList):
                        bigraph_adjList[idx_r] = []
                    if(idx_r != idx_c):             # visits to self are not counted
                        if idx_r in top3K_anchorset_id:
                            bigraph_adjList[idx_r].append(idx_c)


        idx_anchor = no_of_anchors
        while(idx_anchor>0):
            # get list of keys sorted in descending order according to their degree [degree = length of adjlist]
            sorted_adj = sorted(bigraph_adjList, key=lambda k: len(bigraph_adjList[k]), reverse=True)

            # find highest cardinality in visited vertices (keys)
            a_idx = sorted_adj.pop(0)
            score = len(bigraph_adjList[a_idx])
            if(a_idx not in cumulative_score):
                cumulative_score[a_idx] = 0
            
            cumulative_score[a_idx] = cumulative_score[a_idx] + score

            #remove a_idx from visited set (V1) of bipartite graph and corresponding connections from origin vertex sets (V2)
            remove_vids = bigraph_adjList.pop(a_idx)
            for k in bigraph_adjList:
                for v_idx in remove_vids:
                    if(v_idx in bigraph_adjList[k]):
                        bigraph_adjList[k].remove(v_idx)
    
            idx_anchor = idx_anchor - 1

    # sort vertices by their cumulative scores, get top-k anchors
    sorted_cumulative_anchors = sorted(cumulative_score, key = lambda k: cumulative_score[k], reverse=True)

    new_anchorset_id = sorted_cumulative_anchors[:no_of_anchors]
    
    new_anchorset_id=np.reshape(new_anchorset_id, (1, -1)).T
    return new_anchorset_id
    
    
# Get Reachability Counts
def get_dist_max(anchorset_id, dists, device):

    # anchor to node similarity
    dist_max1 = torch.zeros((dists.shape[0],len(anchorset_id))).to(device)
    dist_argmax1 = torch.zeros((dists.shape[0],len(anchorset_id))).long().to(device)

    # node to anchor similarity
    dist_max2 = torch.zeros((dists.shape[0],len(anchorset_id))).to(device)

    dist = torch.from_numpy(dists).float()


    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dist[:, temp_id]
        
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        
        dist_max1[:,i]=dist_max_temp
        dist_argmax1[:,i]=torch.tensor(anchorset_id[i])
        
        ########################################################
        dist_temp2 = dist[temp_id,:]
        dist_max2[:,i] = dist_temp2
        ##########################################################

    return dist_max1, dist_argmax1,dist_max2



# ANCHOR SELECTION FUNCTION
def preselect_anchor(data,bipartite_graph_RW_mat_list,data_dists,args,select_anchors,layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):


    if select_anchors=='random':
        anchorset_id = get_Random__single_anchor_node(data.num_nodes,data_dists,device,c=1)


    if select_anchors=='topK':
        anchorset_id = get_TopK__single_anchor_node(data.num_nodes,data_dists,device,c=1)


    if select_anchors=='DiversifiedRandomK':
        Random3K_anchorset_id = get_Random__single_anchor_node(data.num_nodes,data_dists,device,c=3)

        Random3K_anchorset_id=[l.tolist() for l in Random3K_anchorset_id]
        Random3K_anchorset_id_list = []
        for sublist in Random3K_anchorset_id:
            for item in sublist:
                Random3K_anchorset_id_list.append(item)

        anchorset_id = get_orthoTopK(bipartite_graph_RW_mat_list,Random3K_anchorset_id_list,args,data.num_nodes,data_dists,device,c=1)


    if select_anchors=='DiversifiedTopK':
        Top3K_anchorset_id = get_TopK__single_anchor_node(data.num_nodes,data_dists,device,c=3)

        Top3K_anchorset_id=[l.tolist() for l in Top3K_anchorset_id]
        Top3K_anchorset_id_list = []
        for sublist in Top3K_anchorset_id:
            for item in sublist:
                Top3K_anchorset_id_list.append(item)

        anchorset_id = get_orthoTopK(bipartite_graph_RW_mat_list,Top3K_anchorset_id_list,args,data.num_nodes,data_dists,device,c=1)
        

    data.dists_max, data.dists_argmax,data.dists_max2 = get_dist_max(anchorset_id, data_dists, device)
    return anchorset_id
