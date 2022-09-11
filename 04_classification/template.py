# Author: Íris Friðriksdóttir
# Date: 11.september 2022
# Project: 04_Classification
# Acknowledgements: Kristján Daðason
#

# 1.1 split_Data is not excatcly the same.
# Independent part, Posterior method should be better, takes in more data...

from tools import load_iris, split_train_test

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Independent_data import data_points

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
        print(counts_class[k])

        prob.append(prior_prob)
    print(len(train_features))
    print(f"prior probability: {prob}")
    
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

def create_targets(k: int, classes: np.ndarray):
    listi = []
    counter = 0

    for i in range(k):
        counter +=1
        if counter <= 600:
            listi.append(classes[0])
        if counter > 600 and counter <= 2000:
            listi.append(classes[1])
        if counter > 2000 and counter <= 2500:
            listi.append(classes[2])
    return np.array(listi)

def create_features(features: np.ndarray, targets: np.ndarray):
    var = 0.5
    mean_k0 = mean_of_class(features, targets, 0)
    mean_k1 = mean_of_class(features, targets, 1)
    mean_k2 = mean_of_class(features, targets, 2)
    
    data_k0 = np.random.normal(mean_k0, var, size = (600,4))
    data_k1 = np.random.normal(mean_k1, var, size = (1400,4))
    data_k2 = np.random.normal(mean_k2, var, size = (500,4))

    data = np.append(data_k0, data_k1, axis=0)
    data_pkt = np.append(data, data_k2, axis=0)
    return data_pkt

# from assignment 02

def scatter_plot_points(
    features: np.ndarray,
    pred_tar: np.ndarray,
    targets: np.ndarray,
    method: str
):
    # predicted targets
    # predict_t = knn_predict(points, point_targets, classes, k)
    colors = ['yellow', 'purple', 'blue']
    markers = [mlines.Line2D([],[], color=x, marker='o', linestyle="None") for x in colors]
    counter_wrong = 0
    counter_right = 0
    for i in range(len(targets)):
        [x, y] = features[i,:2]
        if pred_tar[i] == targets[i]:
            plt.scatter(x, y, c=colors[targets[i]], edgecolors='green',
            linewidths=1)
            counter_right += 1
            
        else:
            plt.scatter(x, y, c=colors[targets[i]], edgecolors='red',
            linewidths=1)
            counter_wrong += 1
    print(counter_right)
    print(counter_wrong)

    # plt.legend(title = "Yellow = 0, Purple = 1, Blue = 2")
    #plt.legend(labels = [1, 2, 3])
    plt.legend(markers,colors)
    plt.figtext(0.5,0.01, f"Correct predicts: {counter_right}, Wrong predicts: {counter_wrong}")
    plt.title(method)
    plt.show()


if __name__ == '__main__':
    # PART 1.1
    # Load the data
    # features, targets, classes = load_iris()
    # (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets, train_ratio=0.6)


    # print(len(train_features))
    # print(len(train_targets))
    # print(mean_of_class(train_features, train_targets, 0))
    # print(mean_of_class(train_features, train_targets, 1))
    # print(mean_of_class(train_features, train_targets, 2))


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
    # likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
    # print(predict(likelihoods))
    
    # PART 2.1
    # ---- likelihoods with maximum aposteriori classification ---
    # print(maximum_aposteriori(train_features, train_targets, test_features, classes))

    # PART 2.2
    # ---- Predict with maximum aposteriori classification ---
    # likelihoods_aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
    # print(predict_aposteriori(likelihoods_aposteriori))

    # Accuracy of both methods
    # likelihood_targets = predict(likelihoods)
    # aposteriori_targets = predict_aposteriori(likelihoods_aposteriori)
    # print(f"The accuracy for likelihood probability: {accuracy(likelihood_targets, test_targets)}")
    # print(f"The accuracy for likelihood with Aposteriori method: {accuracy(aposteriori_targets, test_targets)}")

    # confusion matrixes
    # print("Confusion matrix for likelihood method:")
    # print(confusion_mat(likelihood_targets, test_targets))
    # print("Confusion matrix for likelihood Aposteriori:")
    # print(confusion_mat(aposteriori_targets, test_targets))

    # ---- Plot confusion matrixes ----
    # cm_likelihood = confusion_mat(likelihood_targets, test_targets)
    # cm_dp_likelihood = ConfusionMatrixDisplay(cm_likelihood).plot()
    # plt.title("Confusion Matrix for likelihood method")
    # plt.show()

    # cm_aposteriori = confusion_mat(aposteriori_targets, test_targets)
    # cm_dp_ap = ConfusionMatrixDisplay(cm_aposteriori).plot()
    # plt.title("Confusion Matrix for Aposteriori method")
    # plt.show()

# INDEPENDENT PART USING IRIS DATA SET
# create more data points around the mean with gen data
# use gen data from task 3, find mean with mean and cov with function 
#     features, targets, classes = load_iris()

#     # ----- generating data for independent
#     target_pkt = []
#     data_pkt = []

#     target_pkt = create_targets(2500, classes)
#     data_pkt = create_features(features, targets)
#     # -------    
#     (train_features, train_targets), (test_features, test_targets) = split_train_test(data_pkt, target_pkt, train_ratio=0.6)

# # ---- Predict classification based on probability ----
#     likelihoods = maximum_likelihood(train_features, train_targets, test_features, classes)
#     # print(predict(likelihoods))
    
#     # ---- likelihoods with maximum aposteriori classification ---
#     # print(maximum_aposteriori(train_features, train_targets, test_features, classes))

#     # ---- Predict with maximum aposteriori classification ---
#     likelihoods_aposteriori = maximum_aposteriori(train_features, train_targets, test_features, classes)
#     # print(predict_aposteriori(likelihoods_aposteriori))

#     # Accuracy of both methods
#     likelihood_targets = predict(likelihoods)
#     # print(f"The predicted target list for likelihood: {likelihood_targets}")
#     aposteriori_targets = predict_aposteriori(likelihoods_aposteriori)
#     # print(f"The predicted target list for aposteriori: {aposteriori_targets}")
#     print(f"The accuracy for likelihood probability: {accuracy(likelihood_targets, test_targets)}")
#     print(f"The accuracy for likelihood with Aposteriori method: {accuracy(aposteriori_targets, test_targets)}")

#     # confusion matrixes
#     print("Confusion matrix for likelihood method:")
#     print(confusion_mat(likelihood_targets, test_targets))
#     print("Confusion matrix for likelihood Aposteriori:")
#     print(confusion_mat(aposteriori_targets, test_targets))

#     # Plot with scatterplot
#     scatter_plot_points(test_features, likelihood_targets, test_targets, "Likelihood")
#     scatter_plot_points(test_features, aposteriori_targets, test_targets, "Aposteriori")
