import numpy as np
import pandas as pd
import itertools
import math
from scipy.stats import mvn
from scipy.stats import multivariate_normal
import quadpy


def create_box_thresholds(num_features, node_count, features, thresholds):
    # Create an empty list of lists to save the decision boundaries for each feature
    BOX = [[] for i in range(num_features)]
    
    # Get the decision boundaries for each feature
    for i in range(node_count):
        for j in range(num_features):
            if features[i] == j:
                BOX[j].append(thresholds[i])

    # Sort the decision boundaries in ascending order for each feature
    for i in range(num_features):
        BOX[i].sort()

    # Expand the surface by 200 in each direction to cover almost all the probability surface
    for i in range(num_features):
        if BOX[i] != []:
            BOX[i].insert(0,-1000000)
            BOX[i].append(1000000)
        else:
            BOX[i].insert(0,-1000000)
            BOX[i].append(1000000)
    
    return BOX


def boxes_with_labels(num_features, BOX):
    # Number of upper and lower thresholds for the boxes
    box_thresholds = []
    for i in range(num_features):
        box_thresholds.append('B'+str(i+1))
        box_thresholds.append('T'+str(i+1))

    # Number of boxes the classifier has
    num_boxes = 1
    for i in range(len(BOX)):
        num_boxes *= (len(BOX[i])-1)

    index = np.arange(0,num_boxes,1)

    # Create an empty dataframe for all the decision areas; we call them boxes, stemming from 2D examples
    boxes = pd.DataFrame(columns=box_thresholds, index=index)
    
    # Create all boxes
    T = []

    for i in range(len(BOX)):
        T.append([])

    for i in range(len(BOX)):
        for j in range(len(BOX[i])-1):
            T[i].append([BOX[i][j],BOX[i][j+1]])

    Z = list(itertools.product(*T))

    Help = []

    for j in range(len(boxes)):
        join = []
        for i in range(len(BOX)):
            join += Z[j][i]
        Help.append(join)

    boxes = pd.DataFrame(Help, columns=box_thresholds)
    
    return boxes
    
def input_features(test_coordinates, num_features):
    while len(test_coordinates) < num_features:
        try:
            input_features = float(input('Feature values: '))
            test_coordinates.append(input_features)
        except ValueError:
            print('Enter a number.')
            
    return test_coordinates
    
def create_covariance_matrix(num_features):
    A = np.random.rand(num_features, num_features)
    cov = np.dot(A, A.transpose())
    
    return cov
    
def calculate_robustness(coord, covariance_matrix, boxes):
    Tops = list(boxes.columns[1::2])
    Bottoms = list(boxes.columns[0::2])[:-1]
    Tops = boxes[Tops]
    Bottoms = boxes[Bottoms]

    rob_fast = 0

    for i in range(boxes.shape[0]):
        low = Bottoms.iloc[i]
        upp = Tops.iloc[i]

        p_box,i_box = mvn.mvnun(low,upp,coord,covariance_matrix)
        rob_fast += p_box    

    return rob_fast            

def calculate_numerical_robustness(coord, covariance_matrix, boxes, dim):
    val_all = []
    scheme = quadpy.cn.stroud_cn_5_4(dim)

    for i in range(boxes.shape[0]):
        values = boxes.iloc[i][:dim*2]
        my_list = []

        for j in range(0, 2*dim, 2):
            my_list.append(values[j:j+2])
        my_list = tuple(my_list)

        val = scheme.integrate(
            lambda x: multivariate_normal(coord,covariance_matrix).pdf(x.T),
            quadpy.cn.ncube_points(
                *my_list
            ),
        )
        val_all.append(val)

        
    # Eliminate NaNs
    rob = [item for item in val_all if not(math.isnan(item)) == True]
    res = sum(rob) 
    
    return res
