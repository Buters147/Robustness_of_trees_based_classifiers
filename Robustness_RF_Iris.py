import numpy as np
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from Functions import create_box_thresholds
from Functions import boxes_with_labels
from Functions import create_covariance_matrix
from Functions import calculate_robustness

# Some fixed values
seed = 123
deepness = 4
trees = 10 # number of trees for the RF

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Create a Gaussian Classifier
clf = RandomForestClassifier(max_depth=deepness, n_estimators=trees, random_state=seed)
clf.fit(X_train, y_train)

# Fix a test datapoint
coordinates = [1,1,1,1]

# Fix the covariance matrix
cov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

# Classification according to the classifier
label = clf.predict([coordinates])[0]
print('Predicted label:', label)

# Robustness Calculation

# Numbers from the RF
forest_num_features = []
forest_node_count = 0
forest_features = []
forest_thresholds = []

# Get the numbers from the trees
for i in range(trees):
    estimator = clf.estimators_[i]
    
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
for i in range(len(BOX)):
    BOX[i] = list(set(BOX[i]))
    BOX[i].sort()

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
