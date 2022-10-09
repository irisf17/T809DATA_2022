# Author: Íris Friðriksdóttir
# Date: 9. oktober 2022
# Project: 08_SVM
# Acknowledgements: 
#

# NOTE: Your code should NOT contain any main functions or code that is executed
# automatically.  We ONLY want the functions as stated in the README.md.
# Make sure to comment out or remove all unnecessary code before submitting.

# part 1.1
# making small data with blobs.


# part 1.2


# part 1.3


# part 1.4


# part 1.5
# c lowers then more stack variables?


# part 2.1
# we can start using more complicated data set
# 


# part 2.2


# part 2.3


from calendar import c
from tools import plot_svm_margin, load_cancer
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt


def _plot_linear_kernel():
    X,t = make_blobs(n_samples=40, centers=2, random_state=6)
    clf = svm.SVC(C=1000, kernel= 'linear')
    clf.fit(X,t)
    # plot of for 2 points
    # plot_svm_margin(clf, X[0:2,:], t[0:2])
    # print(X[0:2,:])
    plt.title("SVM hyperplane")
    plot_svm_margin(clf, X, t)
    # There are 20 support vectors for each class
    # The decision boundary is linear.
    return X,t


def _subplot_svm_margin(
    svc,
    X: np.ndarray,
    t: np.ndarray,
    num_plots: int,
    index: int
):
    '''
    Plots the decision boundary and decision margins
    for a dataset of features X and labels t and a support
    vector machine svc.

    Input arguments:
    * svc: An instance of sklearn.svm.SVC: a C-support Vector
    classification model
    * X: [N x f] array of features
    * t: [N] array of target labels
    '''
    # similar to tools.plot_svm_margin but added num_plots and
    # index where num_plots should be the total number of plots
    # and index is the index of the current plot being generated
    plt.subplot(1, num_plots, index)
    plt.scatter(X[:, 0], X[:, 1], c=t, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z,
        colors='k', levels=[-1, 0, 1],
        alpha=0.5, linestyles=['--', '-', '--'])

    # plot support vectors
    ax.scatter(
        svc.support_vectors_[:, 0],
        svc.support_vectors_[:, 1],
        s=100, linewidth=1, facecolors='none', edgecolors='k')


def _compare_gamma():
    plt.figure()
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000, kernel= 'rbf', gamma='scale')
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 1)
    plt.title("gamma = default")

    clf = svm.SVC(C=1000, kernel= 'rbf', gamma=0.2)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 2)
    plt.title("gamma = 0.2")

    clf = svm.SVC(C=1000, kernel= 'rbf', gamma=2)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 3, 3)
    plt.title("gamma = 2")
    plt.show()

def _compare_C():
    plt.figure()
    X, t = make_blobs(n_samples=40, centers=2, random_state=6)

    clf = svm.SVC(C=1000, kernel= 'linear')
    clf.fit(X,t)
    print(len(clf.support_vectors_))
    _subplot_svm_margin(clf, X, t, 5, 1)
    plt.title("C = 1000")

    clf = svm.SVC(C=0.5, kernel= 'linear')
    clf.fit(X,t)
    print(len(clf.support_vectors_))
    _subplot_svm_margin(clf, X, t, 5, 2)
    plt.title("C = 0.5")

    clf= svm.SVC(C=0.3, kernel= 'linear')
    clf.fit(X,t)
    print(len(clf.support_vectors_))
    _subplot_svm_margin(clf, X, t, 5, 3)
    plt.title("C = 0.3")

    clf= svm.SVC(C=0.05, kernel= 'linear')
    clf.fit(X,t)
    print(len(clf.support_vectors_))
    _subplot_svm_margin(clf, X, t, 5, 4)
    plt.title("C = 0.05")

    clf= svm.SVC(C=0.0001, kernel= 'linear')
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 5, 5)
    print(len(clf.support_vectors_))
    plt.title("C = 0.0001")    

    plt.show()


def train_test_SVM(
    svc,
    X_train: np.ndarray,
    t_train: np.ndarray,
    X_test: np.ndarray,
    t_test: np.ndarray
):
    '''
    Train a configured SVM on <X_train> and <t_train>
    and then measure accuracy, precision and recall on
    the test set

    This function should return (accuracy, precision, recall)
    '''
    svc.fit(X_test, t_test)
    predicted = svc.predict(X_test)

    acc = accuracy_score(t_test, predicted)
    prec = precision_score(t_test, predicted)
    rec = recall_score(t_test, predicted)
    mean = acc+prec+rec / 3

    print(f"Accuracy score: {acc}")
    print(f"Precision score: {prec}")
    print(f"Recall score: {rec}")
    print(f"MEAN: {mean}")
    return acc, prec, rec


def compare_INDE():
    plt.figure()
    X,t = make_blobs(n_samples=100, centers=2, random_state=6)
    clf = svm.SVC(C=10, kernel= 'rbf', gamma=0.5)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 4, 1)
    plt.title("C = 10, gamma=0.5")

    clf = svm.SVC(C=0.5, kernel= 'rbf', gamma=0.5)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 4, 2)
    plt.title("C = 0.5, gamma=0.5")

    clf= svm.SVC(C=10, kernel= 'rbf', gamma=0.1)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 4, 3)
    plt.title("C = 10, gamma=0.1")

    clf= svm.SVC(C=0.5, kernel= 'rbf', gamma=0.1)
    clf.fit(X,t)
    _subplot_svm_margin(clf, X, t, 4, 4)
    plt.title("C = 0.5, gamma=0.1")

    # clf= svm.SVC(C=0.0001, kernel= 'rbf', gamma=2)
    # clf= svm.SVC(C=10, kernel= 'rbf', gamma=2)
    # clf.fit(X,t)
    # _subplot_svm_margin(clf, X, t, 5, 3)
    # plt.title("C = 10, gamma=2")


    plt.show()

if __name__ == '__main__':
    # Part 1.1
    '''
    _plot_linear_kernel()

    # Part 1.3
    _compare_gamma()

    # Part 1.5
    _compare_C()  
    '''
    # Part 2.1
    '''
    (X_train, t_train), (X_test, t_test) = load_cancer()
    # default kernel = 'rbf'
    svc = svm.SVC(C=1000) 

    train_test_SVM(svc, X_train, t_train, X_test, t_test)

    # Part 2.2
    # Linear
    svc = svm.SVC(C=1000, kernel='linear')
    print("linear")
    train_test_SVM(svc, X_train, t_train, X_test, t_test)
    # radial basis
    print("radial basis")
    svc = svm.SVC(C=1000, kernel='rbf')
    train_test_SVM(svc, X_train, t_train, X_test, t_test)
    # polynomial
    print("polynomial")
    svc = svm.SVC(C=1000, kernel='poly')
    train_test_SVM(svc, X_train, t_train, X_test, t_test)
    '''
    '''
    # Indepentend Section
    # Comparing visually different parameters on a bigger blob dataset
    # Hyper-parameters like C or Gamma control how 
    # wiggling the SVM decision boundary could be.
    # using default kernel = rbf
    compare_INDE()
    '''



