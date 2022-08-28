# Author: Íris Friðriksdóttir
# Date: 20.August 2022
# Project: 01_decision_trees
# Acknowledgements: Bjarki Laxdal, Ignas Sauliusson and Kristján Daðason
#

from operator import length_hint
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
from tools import load_iris, split_train_test


def prior(targets: np.ndarray, classes: list) -> np.ndarray:
 
    # Calculate the prior probability of each class type
    # given a list of all targets and all class types

    N = len(targets)
    if N == 0:
        N=1
    output = []
    counter = 0
    for i in range(len(classes)):
        for k in range(len(targets)):
            if targets[k] == classes[i]:
                counter += 1
        output.append(counter/N)
        counter = 0

    return output

def split_data(
    features: np.ndarray,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    # Split a dataset and targets into two seperate datasets
    # where data with split_feature < theta goes to 1 otherwise 2
    features_1 = []
    features_2 = []
    targets_1 = []
    targets_2 = []
    for i in range(len(features)):
        if features[i, split_feature_index] < theta:
            features_1.append(features[i, split_feature_index])
            targets_1.append(targets[i])
        else:
            features_2.append(features[i, split_feature_index])
            targets_2.append(targets[i])
 
    return (features_1, targets_1), (features_2, targets_2)


def gini_impurity(targets: np.ndarray, classes: list) -> float:

    probability = prior(targets, classes)
    prob_new = []
    for i in range(len(probability)):
        prob_new.append(np.power(probability[i],2))
    gini_val = 1/2 * (1- sum(prob_new))
    
    return gini_val


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2, classes)
    N = len(t1) + len(t2)
    N0 = len(t1)
    N1 = len(t2)

    av_impurity = (N0*g1 /N) + (N1*g2 /N)
    return av_impurity


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    '''
    # Calculate the gini impurity for a split on split_feature_index
    # for a given dataset of features and targets.
    '''
    (f1,t1), (f2,t2) = split_data(features, targets, split_feature_index, theta)
    output = weighted_impurity(t1, t2, classes)
    return output
    

def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    # '''
    index = [0,1,2,3]
    best_gini1 = 1

    for i in range(len(index)):
        # Creating the best threshold
        theta = np.linspace(features[:,i].min(), features[:,i].max(), num_tries+2)[1:-1]
        for t in range(len(theta)):
            best_gini = total_gini_impurity(features, targets,classes, i, theta[t])
            # print(best_gini)
            # print(theta[t])
            if best_gini < best_gini1:
                best_gini1 = best_gini
                best_dim = i
                best_theta = theta[t]
    return best_gini1, best_dim, best_theta


class IrisTreeTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2],
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the Iris dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier()

    def train(self):
        # fit the training data
        train_data = self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        # ---- returns an array of predictions for features  ---
        predict_test_features = self.tree.predict(self.test_features)

        accuracy_test = accuracy_score(self.test_targets, predict_test_features)
        return accuracy_test

    def plot(self):
        plot_tree(self.tree)
        plt.show()
        

    def plot_progress(self):
        # Independent section
        # fit the training data
        accr = []
        for i in range(1, len(self.train_targets)):
            fit_train = self.tree.fit(self.train_features[:i][:],self.train_targets[0:i])
            accuracy_test = self.accuracy()
            accr.append(accuracy_test)
        
        y = range(1, len(self.train_targets))
        plt.plot(y, accr)
        plt.show()
        return accr
  

    def guess(self):
        predict_test_data = self.tree.predict(self.test_features)
        return predict_test_data

    def confusion_matrix(self, predict_targets):
        # Returns confusion matrix on the test data
        return confusion_matrix(self.test_targets, predict_targets)
        
        

if __name__ == '__main__':
    # PART 1.1
    #print(prior([0,0,1], [0,1]))
    #print(prior([0, 2, 3, 3], [0, 1, 2, 3]))

    # PART 1.2
    ## ---- Got help from Ignas Sauliusson to understand the instructions on how to 
    ## ---- build the function.
    # features, targets, classes = load_iris()
    # (f1,t1), (f2,t2) = split_data(features, targets, 2, 4.65)
    # print((f1,t1), (f2,t2))
    # print(len(f1))
    # print(len(f2))
        
    # Part 1.3
    # print(gini_impurity(t1, classes))
    # print(gini_impurity(t2, classes))

    # Part 1.4
    # print(weighted_impurity(t1,t2, classes))

    # Part 1.5
    # print(total_gini_impurity(features, targets, classes, 2, 4.65))

    # Part 1.6
    ## ---- Got a hint from Bjarki Laxdal on how to find the best theta. 
    ## ---- First I tried with some random numbers but never got the same output
    ## ---- as from the instructions. Bjarki gave me a hint on using the numbers from
    ## ---- the features.
    # print(brute_best_split(features, targets, classes, 30))

    # Part 2.1
    # features, targets, classes = load_iris()
    # dt = IrisTreeTrainer(features, targets, classes=classes)
    # dt.train()

    # Part 2.2
    # print(f"The accuracy is: {dt.accuracy()}")

    # Part 2.3
    # dt.plot()

    # Part 2.4
    # print(f"I guessed: {dt.guess()}")
    # print(f"The true targets are: {dt.test_targets}")

    # Part 2.5
    # predict_targets = dt.guess()
    # print(dt.confusion_matrix(predict_targets))

    # Independent part
    ## ---- Got help from Ignas Sauliusson where I was using wrong values for x-axis
    ## ---- minor fix for the values.
    # features, targets, classes = load_iris()
    # dl = IrisTreeTrainer(features, targets, classes=classes, train_ratio=0.6)
    # dl.plot_progress()



