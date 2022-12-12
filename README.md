# Robustness_of_trees_based_classifiers
This repository stores codes to quantify the probabilistic robustness of tree-based classifiers against natural distortions.

The classifiers are trained on the Iris and MNIST dataset. The probabilistic robustness of data samples of tree-based classifiers with a distortion given as a continuous  multivariate probability distributions is computed.

Required Libraries outside of standard libraries:

- Quadpy
- tensorflow
- opencv
- xgboost

Files: 

- Functions.py contains all functions to execute the computations
- Robustness_DT_Iris.py trains a Decision Tree 
- Robustness_DT_Iris_comparison.py trains a Decision Tree and compares the results of the test set when only considering the 99%-CH
- Robustness_DT_MNIST_comparison trains a Decision Tree on the MNIST dataset and compares the result with the 99%-CH and a MC-Sampling approach
- Robustness_RF_Iris.py trains a Random Forest 
- Robustness_RF_MNIST_comparison.py trains a Random Forest on the MNIST dataset and compares the result with the 99%-CH and a MC-Sampling approach
- Robustness_XGB_Iris.py trains a XGboosted Tree 
- Robustness_XGB_Iris_mixed_distrotions.py trains a XGboosted Tree and the distortion is given as a mixed multivariate distribution 
