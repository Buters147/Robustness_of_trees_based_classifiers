import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import itertools
from scipy.stats import chi2
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.stats import weibull_min
from scipy.stats import lognorm
from scipy.stats import mvn

from sklearn.metrics import r2_score

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from Functions import create_box_thresholds
from Functions import boxes_with_labels
from Functions import create_covariance_matrix
from Functions import calculate_robustness

import math

# Some fixed values
seed = 123
deepness = 4
trees = 2 # number of trees (total number of trees will be that times number_of_classes, here 3)
lr = 0.1

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Create a Gaussian Classifier
clf = GradientBoostingClassifier(max_depth=deepness, n_estimators=trees, random_state=seed, learning_rate=lr, verbose=1)
clf.fit(X_train, y_train)

ROB = []
MC = []

for t in range(X_test.shape[0]):
    # Scale and loc parameters to model the data in the dimensions
    # Normal
    loc_1 = X_test[t][0]
    scale_1 = 1

    # Exponential
    loc_2 = X_test[t][1]
    scale_2 = 2

    # Chi2
    loc_3 = X_test[t][2]
    scale_3 = 7
    df_3 = 1

    # lognorm
    loc_4 = X_test[t][3]
    scale_4 = 2
    s_4 = 2
    
    # Test Datapoint
    coordinates = np.array([loc_1,loc_2,loc_3,loc_4])
    
    # Generate datapoints that are multivariate normally distributed with mean 0 and the covariance matrix as determined above
    samples = 1000

    mean = [0,0,0,0]
    cov = np.array([[1,0.1,0.2,0.3],[0.1,1,0.1,0.2],[0.2,0.1,1,0.3],[0.3,0.2,0.3,1]])

    norm_1,norm_2,norm_3,norm_4 = np.random.multivariate_normal(mean, cov, samples).T

    # Transform the data to be uniformly distributed
    unif_1 = norm.cdf(norm_1)
    unif_2 = norm.cdf(norm_2)
    unif_3 = norm.cdf(norm_3)
    unif_4 = norm.cdf(norm_4)

    d_1 = norm.ppf(unif_1, loc=loc_1, scale=scale_1)
    d_2 = expon.ppf(unif_2, loc=loc_2, scale=scale_2)
    d_3 = chi2.ppf(unif_3, df=df_3, loc=loc_3, scale=scale_3)
    d_4 = lognorm.ppf(unif_4, s=s_4, loc=loc_4, scale=scale_4)
    
    # Classification according to the classifier
    label = clf.predict([coordinates])[0]
    print('Predicted label:', label)

    # Numbers from the RF
    forest_num_features = []
    forest_node_count = 0
    forest_features = []
    forest_thresholds = []

    for h in range(trees):
        for j in range(len(clf.estimators_[h])):
            estimator = clf.estimators_[h][j]

            # Get the number of features
            tree_num_features = estimator.tree_.n_features
            forest_num_features.append(tree_num_features)

            # Get the number of nodes
            tree_node_count = estimator.tree_.node_count
            forest_node_count += tree_node_count

            # Get the tree features
            tree_features = estimator.tree_.feature
            forest_features = forest_features + list(tree_features)

            # Get the thresholds
            tree_thresholds = estimator.tree_.threshold
            forest_thresholds = forest_thresholds + list(tree_thresholds)

    forest_num_features = max(forest_num_features)
    forest_thresholds = np.array(forest_thresholds)
    forest_features = np.array(forest_features)
    
    # Get the box thresholds
    BOX = create_box_thresholds(forest_num_features, forest_node_count, forest_features, forest_thresholds)

    # Get all the boxes into a fataframe
    boxes = boxes_with_labels(forest_num_features, BOX)

    # Determine the label for each box (here we subtract 0.01 to make sure we get the right label)
    boxes['class'] = clf.predict(1/2*(boxes.values[:,1::2] + boxes.values[:,:-1:2]))

    # Expand the boxes to infinity in both directions
    boxes = boxes.replace(1000000.0, np.inf)
    boxes = boxes.replace(-1000000.0, -np.inf)

    # Only keep the boxes that have the label of the test point
    boxes = boxes[boxes['class'] == label]

    # DB Transformation
    for h in range(boxes.shape[0]):
        for j in range(boxes.shape[1]-1):
            if j == 1 or j == 0:
                boxes.iloc[[h],[j]] = norm.ppf(norm.cdf(boxes.iloc[h][j], loc=loc_1, scale=scale_1), loc=0, scale=1)
            if j == 3 or j == 2:
                boxes.iloc[[h],[j]] = norm.ppf(expon.cdf(boxes.iloc[h][j], loc=loc_2, scale=scale_2), loc=0, scale=1)
            if j == 5 or j == 4:
                boxes.iloc[[h],[j]] = norm.ppf(chi2.cdf(boxes.iloc[h][j], df=df_3, loc=loc_3, scale=scale_3), loc=0, scale=1)
            if j == 7 or j == 6:
                boxes.iloc[[h],[j]] = norm.ppf(lognorm.cdf(boxes.iloc[h][j], s=s_4, loc=loc_4, scale=scale_4), loc=0, scale=1)

    # Delete boxes that have 0 mass
    delete = []

    for h in range(boxes.shape[0]):
        if boxes.iloc[h]['B1'] == boxes.iloc[h]['T1'] or boxes.iloc[h]['B2'] == boxes.iloc[h]['T2'] or boxes.iloc[h]['B3'] == boxes.iloc[h]['T3'] or boxes.iloc[h]['B4'] == boxes.iloc[h]['T4']:
            delete.append(boxes.index[h])

    boxes.drop(delete, axis=0, inplace=True)
    
    # Only keep the boxes that have the label of the test point
    print('Number of boxes:', boxes.shape[0])

    mean_t = [0,0,0,0]
    
    # Result
    rob = calculate_robustness(mean_t, cov, boxes)
    
    #if math.isnan(rob) == True:
    #    rob = 1
        
    ROB.append(rob)
    print('Robustness of the datapoint ' + '[' + str(loc_1) + ', ' + str(loc_2) + ', ' + str(loc_3) + ', ' + str(loc_4) + ']: ' + str(rob))
    print()
   