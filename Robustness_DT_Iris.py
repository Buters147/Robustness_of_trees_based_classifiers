import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split

from Functions import create_box_thresholds
from Functions import boxes_with_labels
from Functions import create_covariance_matrix
from Functions import calculate_robustness

# Some fixed values
seed = 123
deepness = 4

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Basic Decision Tree Classifier
clf = tree.DecisionTreeClassifier(max_depth=deepness, random_state=seed)
clf = clf.fit(X_train, y_train)

# Test Datapoint
coordinates = np.array([1, 1, 1, 1])

# Covariance Matrix
cov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# Classification according to the classifier
label = clf.predict([coordinates])[0]
print('Predicted label:', label)

# Get the number of features
tree_num_features = clf.tree_.n_features
tree_node_count = clf.tree_.node_count
tree_features = clf.tree_.feature
tree_thresholds = clf.tree_.threshold

# Get the box thresholds
BOX = create_box_thresholds(tree_num_features, tree_node_count, tree_features, tree_thresholds)

# Get all the boxes into a fataframe
boxes = boxes_with_labels(tree_num_features, BOX)

# Determine the label for each box (here we subtract 0.01 to make sure we get the right label)
#boxes['class'] = clf.predict(boxes[list(boxes.columns[1::2])]-0.01)
boxes['class'] = clf.predict(1/2*(boxes.values[:,1::2] + boxes.values[:,:-1:2]))

# Expand the boxes to infinity in both directions
boxes = boxes.replace(1000000.0, np.inf)
boxes = boxes.replace(-1000000.0, -np.inf)

# Only keep the boxes that have the label of the test point
boxes = boxes[boxes['class'] == label]

print('Number of boxes:', boxes.shape[0])

# Result
rob = calculate_robustness(coordinates, cov, boxes)
print('Robustness of the datapoint ' + str(coordinates) + ': ' + str(rob))

import scipy.special as sc

#print(boxes.shape)
r=0
# We take the boxes where 99% of the surface is reached
while (sc.gammainc(tree_num_features/2, r**2/2) < 0.99):
    r+=1

# Eliminate the boxes that are outside of the 99% confidence ellipse
for i in range(tree_num_features):
    boxes = boxes[boxes['T'+str(i+1)]>=coordinates[i]-r*np.sqrt(cov[i][i])]
    boxes = boxes[boxes['B'+str(i+1)]<=coordinates[i]+r*np.sqrt(cov[i][i])]
    
#print(boxes.shape)
print('Number of boxes in 99% confidence ellipse:', boxes.shape[0])

# Result
rob = calculate_robustness(coordinates, cov, boxes)
print('Robustness of the datapoint ' + str(coordinates) + ' in 99% confidence ellipse: ' + str(rob))
