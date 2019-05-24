import torch
import numpy as np
import itertools
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

#import line_profiler

class HenzePenrose():
  #calculates the henze-penrose statistic between set_1 and set_2
  #inputs: two different classes sampled at the same level of a net
  #distance: either euclid or cosine
  #Required Shape: nxitem shape
  #For example: if I have 100 samples of class 1 and 200 of class 2,
  #with each being a 5x5 image, set_1 should be of shape 100x5x5 and
  #set_2 should be of shape 200x5x5. Every dimension after the first should
  #be equivalent.
  #@profile
  def __init__(self, distance_method,device):
    if distance_method == 'euclid':
      self.distance_function = self.euclid
    self.device = device

  #@profile
  def construct_graph(self,n,distances,pairs):
    #constructs the nxn matrix representing the graph
    #distances: ith element represents the distance
    #between the elements specified in the ith entry
    #of pairs
    graph = np.zeros((n,n))

    pairs = np.array(pairs)
    graph[pairs[:,0],pairs[:,1]] = distances

    return graph

  #@profile
  def euclid(self, vectors, batch_size=100000):
    #vectors: nxdim array
    n,dim = vectors.shape

    #list of pairs of points to analyze
    pairs = [i for i in itertools.combinations(range(n),2)]
    comb = len(pairs)
    batches = int(np.ceil(comb/batch_size))
    final_distances = torch.zeros(comb)

    for i in range(batches):
        #reorganize vectors into a batch_sizex2xdim shape
        if (i+1)*batch_size > comb:
            distances = vectors[pairs[i*batch_size:]].to(self.device)
        else:
            distances = vectors[pairs[i*batch_size:(i+1)*batch_size]]

        distances[:,1] *= -1

        #transform distances into a combx1 vector containing
        #the euclidean distances
        distances = torch.sum(distances,dim=1) #combxdim
        distances = distances * distances      #combxdim
        distances = torch.sum(distances,dim=1) #comb

        if (i+1)*batch_size > comb:
            final_distances[i*batch_size:] = distances
        else:
            final_distances[i*batch_size:(i+1)*batch_size] = distances

    return final_distances,pairs

  #@profile
  def minimumSpanningTree(self, vectors):
    #input shape: nxdim
    #output: nxn numpy array with the minimum spanning tree
    #sparse upper matrix-all nonzero entries represent edges in the MST

    #creates a graph with the distances to all of the vectors
    distances,pairs = self.distance_function(vectors)
    graph = self.construct_graph(len(vectors),distances,pairs)

    #uses the scipy csr_matrix and minimum_spanning_tree commands to create the tree
    graph = csr_matrix(graph)
    Tcsr = minimum_spanning_tree(graph)
    Tcsr = Tcsr.toarray()

    return(Tcsr)

  #@profile
  def calculateSfr(self, class_1, class_2):
    #calculates Sfr, number of edges linking disparate classes in MST
    #inputs: class_1 and class_2 both nx? element tensors, where ? is arbitrary but must
    #be the same between class_1 and class_2

    class_1 = class_1.view(len(class_1),-1)
    class_2 = class_2.view(len(class_2),-1)
    combined = torch.cat((class_1,class_2),dim=0) #joins class_1/2 into one long string of vectors

    mst = self.minimumSpanningTree(combined) #minimum spanning tree as nxn matrix
    edge_locations = np.transpose(np.nonzero(mst)) #nx2 elements, each row is two connected vertices
    cutoff_point = len(class_1)

    #transforms edge_locations to find the number of transition edges
    edge_locations = 1*(edge_locations>=cutoff_point)
    edge_locations = np.abs(edge_locations[:,0]-edge_locations[:,1])

    return(np.sum(edge_locations))

  #@profile
  def __call__(self, class_1, class_2):
    #inputs: class_1 and class_2 both nx? element tensors, where ? is arbitrary but must
    #be the same between class_1 and class_2
    #output: a scalar value
    Sfr = self.calculateSfr(class_1, class_2)
    N = len(class_1)+len(class_2)

    Hxy = 1 - 2*min(N/2,Sfr)/N

    return(Hxy)

#class1 = torch.Tensor(np.load('Transformed Data/train0.npy'))[:100]
#class2 = torch.Tensor(np.load('Transformed Data/train1.npy'))[:100]
#
#hp = HenzePenrose('euclid')
#a = hp.calculateHP(class1,class2)


"""
hp = HenzePenrose('euclid')
v1 = torch.Tensor(
  [[1,1],
  [3,3],
  [5,5],
  [7,7]])
v2 = torch.Tensor(
  [[2,2],
  [4,4],
  [6,6],
  [8,8]])
x = hp.calculateSfr(v1,v2)
a = hp.calculateHP(v1,v2)
"""
