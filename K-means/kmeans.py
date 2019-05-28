import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import random
import os
import copy

def euclidean_distance(x,y):
    try:
        distance = 0
        # pdb.set_trace()
        # print(x.shape)
        for i in range(x.shape[0]):
            distance += (x[i] - y[i])**2
        return distance
    except Exception as e:
        print("error is ", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print("i is ", i)
        print("j is ", j)
        pdb.set_trace()


def center(cluster):
    summation = 0
    for i in range(len(cluster)):
        summation += cluster[i]
    return (summation/len(cluster))

def SSE(clusters, centers):
    summation = 0
    try:
        for i in range (len(clusters)): 
            for j in range (len(clusters[i])):
                # pdb.set_trace()
                summation += euclidean_distance(centers[i], clusters[i][j])
        return summation
    except Exception as e:
        print("error is ", e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        pdb.set_trace()



input = np.loadtxt("p4-data.txt", delimiter = ',')
# input = np.genfromtxt("p4-data.txt",  unpack = True,  delimiter = '\n')
k = int(sys.argv[1])
index = random.randint(0,input.shape[0])
centers = np.asarray([input[index]])
input = np.delete(input, index, 0)

for i in range(k-1):
    index = random.randint(0,input.shape[0])
    centers = np.append(centers,[input[index]], axis=0)
    input = np.delete(input, index, 0)

# clusters = np.asarray([[centers[0]]])
seeds = [[centers[0]]]

for i in range(1,k):
    seeds.append([centers[i]])
# pdb.set_trace()

#CHANGE THE OUTER ARRAY FROM NP.ARRAY TO LIST
# for i in range(1,k):
    # clusters = np.append(clusters,[[centers[i]]], axis=0)
# pdb.set_trace()
index = 0
classification = 0
loss = None
convergence = 0

try:
    while(convergence == 0):
        clusters = copy.deepcopy(seeds)        # initialize the clusters
        for i in range(input.shape[0]):
            distance = 99999999
            new_index = 0
            for j in range(centers.shape[0]):     # Compute distance from the center of each cluster to the point
                eval_dist = euclidean_distance(centers[j],input[i])
                if (eval_dist < distance): # update if distance is smaller
                    distance = eval_dist
                    new_index = j
            clusters[new_index].append(input[i])                # GOAL: [[[1,2],[3,4]],[[5,6],[7,8]]]
        # pdb.set_trace()
        calc_loss = SSE(clusters, centers)
        if (type(loss) != type(None) and loss <= calc_loss):
            convergence = 1
        else:
            print ("Next Iteration, loss is ", loss)
            loss = calc_loss
        for i in range (len(centers)):
            centers[i] = center(clusters[i])

except Exception as e:
    print("error is ", e)
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
    pdb.set_trace()


# convergence = 0

pdb.set_trace()
print(input.shape)