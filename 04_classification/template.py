# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

# 1.1 split_Data is not excatcly the same.
# Independent part, Posterior method should be better, takes in more data...

from tools import load_iris, split_train_test

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def mean_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the mean of a selected class given all features
    and targets in a dataset
    '''
    features_for_class = []
    # mean_class = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            features_for_class.append(features[i])

    # mean_f1 = np.mean(list(zip(*features_of_class))[0])
    # mean_f2 = np.mean(list(zip(*features_of_class))[1])
    # mean_f3 = np.mean(list(zip(*features_of_class))[2])
    # mean_f4 = np.mean(list(zip(*features_of_class))[3])
    # mean_class.append(mean_f1)
    # mean_class.append(mean_f2)
    # mean_class.append(mean_f3)
    # mean_class.append(mean_f4)
    # Axis=0 calculates mean of 1-4 columns in features...
    return np.mean(features_for_class, axis=0)



def covar_of_class(
    features: np.ndarray,
    targets: np.ndarray,
    selected_class: int
) -> np.ndarray:
    '''
    Estimate the covariance of a selected class given all
    features and targets in a dataset
    '''
    features_for_class = []
    # mean_class = []
    for i in range(len(targets)):
        if targets[i] == selected_class:
            features_for_class.append(features[i])
    
    # rowvar = false , each column represents a variable, while the rows contain observations.
    return np.cov(features_for_class, rowvar=False)


def likelihood_of_class(
    feature: np.ndarray,
    class_mean: np.ndarray,
    class_covar: np.ndarray
) -> float:
    '''
    Estimate the likelihood that a sample is drawn
    from a multivariate normal distribution, given the mean
    and covariance of the distribution.
    '''
    return multivariate_normal(mean=class_mean, cov=class_covar).pdf(feature)


def maximum_likelihood(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum likelihood for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    likelihoods = []

    for class_label in classes:
        # means and covs are 3x4
        means.append(mean_of_class(train_features, train_targets, classes[class_label]))
        covs.append(covar_of_class(train_features, train_targets, classes[class_label]))
    
    for i in range(test_features.shape[0]):
        vektor = []
        for j in range(len(classes)):
            vektor.append(likelihood_of_class(test_features[i,:], means[j], covs[j]))
        likelihoods.append(vektor)

    return np.array(likelihoods)

def predict(likelihoods: np.ndarray):
    '''
    Given an array of shape [num_datapoints x num_classes]
    make a prediction for each datapoint by choosing the
    highest likelihood.

    You should return a [likelihoods.shape[0]] shaped numpy
    array of predictions, e.g. [0, 1, 0, ..., 1, 2]
    '''
    # highest value in likelihoods is the best prediction
    return np.argmax(likelihoods, axis=1)

def count_targ(train_targets: np.ndarray):
    counter0 = 0
    counter1 = 0
    counter2 = 0

    for value in train_targets:
        if value == 0:
            counter0 += 1
        if value == 1:
            counter1 += 1
        if value == 2:
            counter2 += 1
    counts_tar = [counter0, counter1, counter2]

    return counts_tar

def maximum_aposteriori(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    classes: list
) -> np.ndarray:
    '''
    Calculate the maximum a posteriori for each test point in
    test_features by first estimating the mean and covariance
    of all classes over the training set.

    You should return
    a [test_features.shape[0] x len(classes)] shaped numpy
    array
    '''
    means, covs = [], []
    likelihoods = []
    prob = []

    for class_label in classes:
        # means and covs are 3x4
        means.append(mean_of_class(train_features, train_targets, classes[class_label]))
        covs.append(covar_of_class(train_features, train_targets, classes[class_label]))
    
    counts_class = count_targ(train_targets)

    for k in range(len(counts_class)):
        prior_prob = counts_class[k]/len(train_features) 
        prob.append(prior_prob)
    
    for i in range(test_features.shape[0]):
        vektor = []
        for j in range(len(classes)):
            vektor.append((likelihood_of_class(test_features[i,:], means[j], covs[j]))*prob[j])
        likelihoods.append(vektor)

    return np.array(likelihoods)

def predict_aposteriori(maximum_aposteriori: np.ndarray):
    # highest value in likelihoods is the best prediction
    return np.argmax(maximum_aposteriori, axis=1)

def accuracy(likelihood_targets: np.ndarray, test_targets: np.ndarray):
    counter = 0
    for i in range(len(test_targets)):
        if likelihood_targets[i] == test_targets[i]:
            counter += 1
    acc = counter/len(test_targets)
    return acc

def confusion_mat(method_targets: np.ndarray, test_targets: np.ndarray):
    # Returns confusion matrix on the train_targets
    # confusion_matrix(true, predicted)
    # true left side down, predicted above the matrix
    return confusion_matrix(test_targets, method_targets)

if __name__ == '__main__':
    # PART 1.1
    # Load the data
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)
    # print(len(train_features))
    # print(len(train_targets))
    # print(mean_of_class(train_features, train_targets, 0))

    # PART 1.2
    # print(covar_of_class(train_features, train_targets, 0))

    # PART 1.3
    # print(test_features)
    # class_mean = mean_of_class(train_features, train_targets, 0)
    # class_cov = covar_of_class(train_features, train_targets, 0)
    # print(likelihood_of_class(test_features[0, :], class_mean, class_cov))

    # PART 1.4
    # -----Likelihood classification based on probability ----
    # print(maximum_likelihood(train_features, train_targets, test_features, classes))

    # PART 1.5
    # ---- Predict classification based on probability ----
    likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    print(predict(likelihoods))
    
    # PART 2.1
    # ---- likelihoods with maximum aposteriori classification ---
    # print(maximum_aposteriori(train_features, train_targets, test_features, classes))

    # PART 2.2
    # ---- Predict with maximum aposteriori classification ---
    likelihoods_aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
    print(predict_aposteriori(likelihoods_aposteriori))

    # Accuracy of both methods
    likelihood_targets = predict(likelihoods)
    aposteriori_targets = predict_aposteriori(likelihoods_aposteriori)
    print(f"The accuracy for likelihood probability: {accuracy(likelihood_targets, test_targets)}")
    print(f"The accuracy for likelihood with Aposteriori method: {accuracy(aposteriori_targets, test_targets)}")

    # confusion matrixes
    print("Confusion matrix for likelihood method:")
    print(confusion_mat(likelihood_targets, test_targets))
    print("Confusion matrix for likelihood Aposteriori:")
    print(confusion_mat(aposteriori_targets, test_targets))

    # ---- Plot confusion matrixes ----
    cm_likelihood = confusion_mat(likelihood_targets, test_targets)
    cm_dp_likelihood = ConfusionMatrixDisplay(cm_likelihood).plot()
    plt.title("Confusion Matrix for likelihood method")
    plt.show()

    cm_aposteriori = confusion_mat(aposteriori_targets, test_targets)
    cm_dp_ap = ConfusionMatrixDisplay(cm_aposteriori).plot()
    plt.title("Confusion Matrix for Aposteriori method")
    plt.show()



