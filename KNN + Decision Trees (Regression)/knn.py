import numpy as np
import sys
# import matplotlib.pyplot as plt
from numpy import genfromtxt
import pdb


def main():
  print("")
  x = genfromtxt("./p2-data/knn_train.csv", delimiter=',')

  print(x.shape)
  
  #load testing data
  y = x[:,0]
  # y = x[:,x.shape[1]-1]
  x = x[:,:x.shape[1]-1]
  x = x[:,1:]  # First : indicates for every row (before the ",") and everything after the comma indicated which columns are to be fetched
  # https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

  # normalize data
  x = (x - np.min(x))/np.ptp(x)

  # tx = genfromtxt(sys.argv[2], delimiter=',')
  dist = []
  neighbours = []
  for i in range(x.shape[0]):
    if(i != x.shape[0]):
      dist.append(euclidean_distance(x[i],np.append(x[:i],x[i+1:], axis = 0)))
      # pdb.set_trace()
      neighbours.append([dist[i][0],dist[i][1],dist[i][2],dist[i][3],dist[i][4]])    # K=4 nearest neighbours
    else:
      dist.append(euclidean_distance(x[i],x[:i]))
      neighbours.append([dist[i][0],dist[i][1],dist[i][2],dist[i][3],dist[i][4]])    # K=4 nearest neighbours

  pdb.set_trace()
  # ty = tx[:,tx.shape[1]-1]
  # tx = tx[:,:tx.shape[1]-1]


def euclidean_distance (x,all_x):
  #include code from
  dist = []
  for i in range(all_x.shape[0]):
    dist.append(np.sqrt(np.dot(np.transpose(x-all_x[i]),(x-all_x[i]))))
  # pdb.set
  # _trace() 
  dist.sort()
  return dist



main()



  