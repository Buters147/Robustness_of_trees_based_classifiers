from tensorflow.keras.datasets import mnist
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score

from Functions import create_box_thresholds
from Functions import boxes_with_labels
from Functions import create_covariance_matrix
from Functions import calculate_robustness

from tqdm import tqdm
from scipy.stats import multivariate_normal
from datetime import datetime

# Load the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape the images
# Training Data
train_data = []

for img in X_train:
    resized_img = cv2.resize(img,(5,5))
    train_data.append(resized_img)
    
X_train = np.array(train_data)

# Test Data
test_data = []

for img in X_test:
    resized_img = cv2.resize(img,(5,5))
    test_data.append(resized_img)
    
X_test = np.array(test_data)

RESHAPED = 25

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.flatten()
y_test = y_test.flatten()

# normalize the datasets
X_train /= 255.
X_test /= 255.

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Some fixed values
seed = 123
deepness = 3
trees = 5 # number of trees for the RF

# Create a RF
clf = RandomForestClassifier(max_depth=deepness, n_estimators=trees, random_state=seed)
clf.fit(X_train, y_train)

import scipy.special as sc

ROB = []
ROB99 = []

#for i in range(len(X_test)):
for i in range(10):
    # Test Datapoint
    coordinates = X_test[i]

    # Covariance Matrix
    L = []
    for k in range(RESHAPED):
        inner = []
        for j in range(RESHAPED):
            if k == j:
                inner.append(X_train[:,i].std())
            else:
                inner.append(0)
        L.append(inner)

    cov = np.array(L)
    
    # Classification according to the classifier
    label = clf.predict([coordinates])[0]
    #print('Predicted label:', label)

    # Numbers from the RF
    forest_num_features = []
    forest_node_count = 0
    forest_features = []
    forest_thresholds = []

    # Get the numbers from the trees
    for a in range(trees):
        estimator = clf.estimators_[a]

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

    # Remove duplicates from the lists and sort them again in ascending order
    for a in range(len(BOX)):
        BOX[a] = list(set(BOX[a]))
        BOX[a].sort()

    # Get all the boxes into a fataframe
    boxes = boxes_with_labels(forest_num_features, BOX)

    # Determine the label for each box (here we subtract 0.01 to make sure we get the right label)
    #boxes['class'] = clf.predict(boxes[list(boxes.columns[1::2])]-0.01)
    boxes['class'] = clf.predict(1/2*(boxes.values[:,1::2] + boxes.values[:,:-1:2]))

    # Expand the boxes to infinity in both directions
    boxes = boxes.replace(1000000.0, np.inf)
    boxes = boxes.replace(-1000000.0, -np.inf)

    # Only keep the boxes that have the same label as the test point
    boxes = boxes[boxes['class'] == label]

    print('Number of boxes:', boxes.shape[0])
    
    # Time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start =", dt_string)
    
    rob = calculate_robustness(coordinates, cov, boxes)
    
    # Time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End =", dt_string)

    print('Robustness of the datapoint ' + str(coordinates) + ': ' + str(rob))
    ROB.append(rob)
    
    #print(boxes.shape)
    r=0
    # We take the boxes where 99% of the surface is reached
    while (sc.gammainc(tree_num_features/2, r**2/2) < 0.99):
        r+=1

    # Eliminate the boxes that are outside of the 99% confidence ellipse
    for a in range(tree_num_features):
        boxes = boxes[boxes['T'+str(a+1)]>=coordinates[a]-r*np.sqrt(cov[a][a])]
        boxes = boxes[boxes['B'+str(a+1)]<=coordinates[a]+r*np.sqrt(cov[a][a])]

    #print(boxes.shape)
    print('Number of boxes in 99% confidence ellipse:', boxes.shape[0])
    
    # Time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start =", dt_string)
    
    # Result
    rob = calculate_robustness(coordinates, cov, boxes)
    
    # Time
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("End =", dt_string)
    
    print('Robustness of the datapoint ' + str(coordinates) + ' in 99% confidence ellipse: ' + str(rob))
    ROB99.append(rob)
    
    print()

R2 = r2_score(ROB, ROB99)
print('R2:', R2)

plt.xlabel('Robustness of the test data samples')
plt.ylabel('Robustness of the test data samples (99% hyperellipsoid)')
plt.plot([min(ROB),max(ROB)],[min(ROB99),max(ROB99)], c='red', linewidth=0.5, linestyle='dashed')
plt.scatter(ROB,ROB99, c='blue', s=10)
plt.title('Robustness Comparison')
plt.show()



