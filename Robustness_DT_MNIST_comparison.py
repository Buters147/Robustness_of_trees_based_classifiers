from tensorflow.keras.datasets import mnist
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
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
# Some fixed values
seed = 123
deepness = 5

# Basic Decision Tree Classifier
clf = tree.DecisionTreeClassifier(max_depth=deepness, random_state=seed)
clf = clf.fit(X_train, y_train)

import scipy.special as sc

# Get the number of features
tree_num_features = clf.tree_.n_features
tree_node_count = clf.tree_.node_count
tree_features = clf.tree_.feature
tree_thresholds = clf.tree_.threshold


ROB = []

for i in range(10):
    coordinates = X_test[i]
    #print(coordinates)

    # Covariance Matrix
    L = []
    for k in range(RESHAPED):
        inner = []
        for j in range(RESHAPED):
            if k == j:
                inner.append(0.0001)
            else:
                inner.append(0)
        L.append(inner)

    cov = np.array(L)

    # Classification according to the classifier
    label = clf.predict([coordinates])[0]
    #print(label)
    
    BOX = create_box_thresholds(tree_num_features, tree_node_count, tree_features, tree_thresholds)
    
    # Remove duplicates from the lists and sort them again in ascending order
    for i in range(len(BOX)):
        BOX[i] = list(set(BOX[i]))
        BOX[i].sort()

    # Get the 99% confidence hyperellipsoid
    r=0
    # We take the boxes where 99% of the surface is reached
    while (sc.gammainc(tree_num_features/2, r**2/2) < 0.99):
        r+=1
        
    # Get the boxes in the 99%-CH
    BOX_ALL = []

    for j in range(tree_num_features):
        B = []
        if len(BOX[j]) > 2:
            for k in range(len(BOX[j])-1):
                if (BOX[j][k]<=coordinates[j]+r*np.sqrt(cov[j][j]) and BOX[j][k+1]>=coordinates[j]-r*np.sqrt(cov[j][j])):
                    B.append(BOX[j][k])
                if (BOX[j][k]<=coordinates[j]+r*np.sqrt(cov[j][j]) and BOX[j][k+1]>=coordinates[j]+r*np.sqrt(cov[j][j])):
                    B.append(BOX[j][k+1])
        else:
            for k in range(len(BOX[j])):
                B.append(BOX[j][k])

        BOX_ALL.append(B)
        
    # Get all the boxes into a fataframe
    boxes = boxes_with_labels(tree_num_features, BOX_ALL)

    # Determine the label for each box (here we subtract 0.01 to make sure we get the right label)
    boxes['class'] = clf.predict(1/2*(boxes.values[:,1::2] + boxes.values[:,:-1:2]))
    
    # Expand the boxes to infinity in both directions
    boxes = boxes.replace(1000000.0, np.inf)
    boxes = boxes.replace(-1000000.0, -np.inf)

    # Only keep the boxes that have the label of the test point
    boxes = boxes[boxes['class'] == label]

    print('Number of boxes:', boxes.shape[0])

    # Result
    rob99 = calculate_robustness(coordinates, cov, boxes)
    print('Robustness in 99% confidence ellipse: ' + str(rob99))
    ROB.append(rob99)
    print()
    
    
# Monte Carlo Sampling
def compute_bf_robustness(model, X, cov_scaled):
    robustness_bruteforce = []
    for x in tqdm(X):
        # generate alternative testpoints for this point
        x_perts = multivariate_normal.rvs(mean=x, cov=cov_scaled, size=n)
        y_pred = model.predict([x]).squeeze()
        y_perts = model.predict(x_perts)
        assert (len(y_perts) == n)
        # compute fraction of cases that are now classified differently
        n_different = sum(y_perts != y_pred)
        frac_different = n_different / n
        robustness_bruteforce.append(1 - frac_different)
    return robustness_bruteforce
    
    
n = 1000000
ROB_BF = compute_bf_robustness(clf, X_test[0:10], cov)
ROB_BF

R2 = r2_score(ROB, ROB_BF)
print('R2:', R2)


plt.xlabel('Real-world robustness')
plt.ylabel('Monte-Carlo Robustness')
plt.plot([min(ROB),max(ROB)],[min(ROB_BF),max(ROB_BF)], c='red', linewidth=0.5, linestyle='dashed')
plt.scatter(ROB,ROB_BF, c='blue', s=10)
plt.title('Robustness Comparison')
plt.show()



