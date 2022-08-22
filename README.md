# Robustness_of_trees_based_classifiers
This repository stores code to compute the real-world-robustness of tree-based classifiers.

The classifiers are trained on the Iris dataset and the real-world robustness of data samples with a distortion given as a multivariate normal distribution with a covariance matrix with 1s on the main diagonal and 0s on the off-diagonals is computed.

- Robustness_DT_Iris.py trains a Decision Tree 
- Robustness_RF_Iris.py trains a Random Forest 
- Robustness_XGB_Iris.py trains a XGboosted Tree 
- Robustness_DT_Iris_comparison.py trains a Decision Tree and compares the results of the test set
- Robustness_DT_MNIST_comparison trains a Decision Tree on the MNIST dataset and compares the result for real-world-robustness with the 99%-CH and the MC-Sampling approach
