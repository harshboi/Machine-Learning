import numpy as np
import sys
# import matplotlib.pyplot as plt
from numpy import genfromtxt
import pdb
import copy
import math

def main():
  print("")
  x = genfromtxt("./p2-data/knn_train.csv", delimiter=',')

  print(x.shape)
  
  #load testing data
  y = x[:,0]
  # y = x[:,x.shape[1]-1]
  # x = x[:,:x.shape[1]-1]

  # x = x[:,1:]  # First : indicates for every row (before the ",") and everything after the comma indicated which columns are to be fetched
  # https://machinelearningmastery.com/index-slice-reshape-numpy-arrays-machine-learning-python/

  # normalize data
  # x = (x - np.min(x))/np.ptp(x)
  sorted_data = copy.deepcopy(x)
  gain, threshold, feature_number, row_num, left, right = thresh(x)
  tree = Tree(create_tree(feature_number, row_num, threshold, left, right))
  node = tree.root
  pdb.set_trace()
  create_regression_tree (1, left, node)
  # create_regression_tree (1, left, node)
  pdb.set_trace()

  # sorted_data = np.sort
  
def create_regression_tree (depth, x, parent):
  gain, threshold, feature_number, row_num, left, right = thresh(x)
  node = (create_tree(feature_number, row_num, threshold, left, right))
  parent.child.append(node)  

  if depth+1 == 5:
    return  node
  else:
    depth += 1
  pdb.set_trace()
  aa = create_regression_tree(depth, left, node)
  bb = create_regression_tree(depth, right, node)
  return parent

  
  

# def stump(x,y):
#   print

def entropy (branch):
  branch_pos = (branch[branch[:,1] == 1]).shape[0]
  branch_neg = (branch[branch[:,1] == -1]).shape[0]
  if (branch_pos/branch.shape[0] == 0 or branch_neg/branch.shape[0] == 0):
    return 0
  try:
    entropy = - (branch_pos/branch.shape[0]) * math.log((branch_pos/branch.shape[0]),2) - (branch_neg/branch.shape[0]) * math.log((branch_neg/branch.shape[0]),2)
    return entropy
  except:
    pdb.set_trace()
 

def info_gain (thresholds, x):
  
  left = x[x[:,0] <= thresholds[0]]     # is the left_child
  right = x[x[:,0] > thresholds[0]]     # is the right child
  # pdb.set_trace()
  left_p = left.shape[0]/x.shape[0]
  right_p = right.shape[0]/x.shape[0]
  # left_branch_pos = (left[left[:,1] == 1]).shape[0]
  # left_branch_neg = (left[left[:,1] == -1]).shape[0]
  # right_branch_pos = (right[right[:,1] == 1]).shape[0]
  # right_branch_neg = (right[right[:,1] == -1]).shape[0]
  
  gain = entropy(x) - left_p * entropy(left) - right_p * entropy(right)
  
  # o_right
  # pdb.set_trace()
  return left, right, gain

# def return_split_data (x, row, threshold):
#   z = copy.deepcopy(x[:,[row,0]])
#   left = x[x[:,0] <= threshold]
#   right = x[x[:,0] > threshold]




def thresh (x):
  max_gain = -1
  max_thresh = -1
  index = -1
  row_num = -1
  max_left = None
  max_right = None
  outputs = []
  hinges = None
  for i in range(1,x.shape[1]):     # i is the column/feature number
    # z = copy.deepcopy(np.append([x[:,i]],[x[:,0]],axis=0))    
    z = copy.deepcopy(x[:,[i,0]])           # 2 columns, feature column and the result/output column, the first colum is the output column
    z = z[z[:,1].argsort()]
    
    # max_thresh = np.median(z,axis=0)[0]
    available_threshholds = []
    try:
      for j in range(z.shape[0]-1):    # j is the row number for the dataset
        #https://people.csail.mit.edu/dsontag/courses/ml16/slides/lecture11.pdf - For the split
        if (z[j][1] != z[j+1][1]):
          available_threshholds.append(z[j][0]+(z[j+1][0] - z[j][0])/2)
          # print("i is ", i, j)
          left, right, gain = info_gain(available_threshholds,z)
          # pdb.set_trace()
          outputs.append(gain)
          # outputs.append((gain,z[j][0]+(z[j+1][0] - z[j][0])/2))
          if (gain > max_gain):
            row_num = j
            index = i
            max_gain = gain
            max_thresh = (z[j][0]+(z[j+1][0] - z[j][0])/2)
            max_left = left
            max_right = right
        else:
          continue
    except:
      print("Stuck inside tresh function")
      pdb.set_trace()
      print("what is a label")
  return(max_gain, max_thresh, index, row_num, max_left, max_right)


# Is the sigmoid function
def sigmoid(w,xi):
  denominator = 1/(1 + np.exp(-np.dot(w,xi)))
  # print (1/denominator)   # is a vector of 256
  return (denominator)

#calculate log likelihood for loss
def loss(x,w,y):
  loss = 0
  #calculate for each example
  for i in range(x.shape[1]):
    loss += y[i] * np.log10(sigmoid(np.transpose(w),x[i])) 
    loss += (1 - y[i]) * np.log(1 - sigmoid(np.transpose(w), x[i]))
  return -loss

def create_tree (feature_number, val, threshold, left, right):
  return Node(feature_number, val, threshold, left, right, next)


# def effectiveness(depth, threshold, input, dt, feature_number):
#   if (depth == 1):
#     # left = x[x[:,0] <= threshold]
#     # right = x[x[:,0] > threshold]
#     if input[feature_number] <= threshold:
#       input.

def append_node (node, feature_number, val, threshold, left = None, right = None):
  node.next.append(Node(feature_number, val, threshold, left = None, right = None));



class Node():
  def __init__(self, feature_number, val, threshold, left, right, child =[]):
    self.feature_number = feature_number   # column number
    self.val = val    # row number
    self.next = [next]  # is the children
    self.left = left    # is how the data is split inside the node
    self.right = right  # is how the data is split inside the node
    self.itself = np.append(left,right, axis=0) # is the original node itslef, NOTE: ordering is different from the dataset for the rows
    self.child = []


class Tree():
  def __init__(self, root):
    self.root = root



main()



  