# Author: Íris Friðriksdóttir
# Date: 22.september 2022
# Project: 05_Backprop
# Acknowledgements: Bjarki Laxdal
#

from cmath import nan
from distutils.command.build_scripts import first_line_re
from turtle import back
from typing import Union
import numpy as np

from tools import load_iris, split_train_test
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# hluti 1.1
#  if a<-100 skila út 0, því annars fer fallið í overflow

# 1.2
# skila út a og z from neurons

# 1.3
# M number of nodes in layer, can find out D from number of x, W1 og W2 are weights from layer 1 and 2
#   random.seed NEEDS TO BE ABOVE split_data

# 1.4
# calcluate the deltas, begin at output/end and go back.

# 2.1
# misclassification shows 500 points.
# are getting better because the values are lowering.+

# 2.2
# testing with test data only need to go forward not back again. 

def sigmoid(x: float) -> float:
    '''
    Calculate the sigmoid of x
    '''
    if x < -100:
        return 0.0

    sigmoid = 1/(1+np.exp(-x))
    return sigmoid


def d_sigmoid(x: float) -> float:
    '''
    Calculate the derivative of the sigmoid of x.
    '''
    sigmoid_d = np.exp(-x)/(np.power((1+np.exp(-x)),2))
    return sigmoid_d


def perceptron(
    x: np.ndarray,
    w: np.ndarray
) -> Union[float, float]:
    '''
    Return the weighted sum of x and w as well as
    the result of applying the sigmoid activation
    to the weighted sum
    '''
    return (np.dot(x,w), sigmoid(np.dot(x,w)))


def ffnn(
    x: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Computes the output and hidden layer variables for a
    single hidden layer feed-forward neural network.
    '''
    a1 = np.array([])
    z1 = np.array([1.0])
    a2 = np.array([])
    y = np.array([])
   
    z0 = np.insert(x,0,1)

    for i in range(M):
        a1 = np.append(a1,perceptron(z0, W1[:,i])[0])
        z1 = np.append(z1,perceptron(z0, W1[:,i])[1])

    for i in range(K):
        a2 = np.append(a2, perceptron(z1, W2[:,i])[0])
        y = np.append(y, perceptron(z1, W2[:,i])[1])
    return (y, z0, z1, a1, a2)


def backprop(
    x: np.ndarray,
    target_y: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Perform the backpropagation on given weights W1 and W2
    for the given input pair x, target_y
    '''
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    d_k = y - target_y
    d_j = []
    dE1 = np.zeros(W1.shape)
    dE2 = np.zeros(W2.shape)

    for j in range(M):
        d_j.append(d_sigmoid(a1[j])*np.dot(W2[j+1], d_k))

    for line in range(dE1.shape[0]):
        for col in range(dE1.shape[1]):
            dE1[line][col] = d_j[col]*z0[line]
    
    for line in range(dE2.shape[0]):
        for col in range(dE2.shape[1]):
            dE2[line][col] = d_k[col]*z1[line]

    return y, dE1, dE2



def train_nn(
    X_train: np.ndarray,
    t_train: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray,
    iterations: int,
    eta: float
) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    # Train a network by:
    # 1. forward propagating an input feature through the network
    # 2. Calculate the error between the prediction the network
    # made and the actual target
    # 3. Backpropagating the error through the network to adjust
    # the weights.
    '''
    W1tr = np.zeros(W1.shape)
    W2tr = np.zeros(W2.shape)
    N = len(X_train)

    Etotal = np.zeros(iterations)
    miscl_rate = np.zeros(iterations)
    last_quesses = np.zeros(iterations)
    # changing train values to hot encoding
    target_y = np.eye(K)[t_train] 
    # print(target_y)

    for i in range(iterations):
        error = 0
        dE1_total = np.zeros(W1.shape)
        dE2_total = np.zeros(W2.shape)
        y_total = np.zeros((N,K))
        for j in range(N):
            y_out, de1_out, de2_out = backprop(X_train[j], target_y[j], M, K, W1, W2)
            y_total[j] = y_out
            dE1_total += de1_out
            dE2_total += de2_out
            error += target_y[j]*np.log(y_out) + (1-target_y[j])*np.log(1-y_out)
        W1 -= eta * dE1_total/N
        W2 -= eta * dE2_total/N
        Etotal[i] = np.sum(-error)/N
        miscl_rate[i] = np.sum(np.argmax(y_total, axis=1) != np.argmax(target_y, axis=1))/N
        last_guesses = np.argmax(y_total, axis=1)

    W1tr = W1
    W2tr = W2
    
    return W1tr, W2tr, Etotal, miscl_rate, last_guesses

def test_nn(
    X: np.ndarray,
    M: int,
    K: int,
    W1: np.ndarray,
    W2: np.ndarray
) -> np.ndarray:
    '''
    Return the predictions made by a network for all features
    in the test set X.
    '''
    N = len(X)
    guesses = np.zeros(N)
    for i in range(N):
        y, z0, z1, a1, a2 = ffnn(X[i], M, K, W1, W2)
        guesses[i] = np.argmax(y)
    return guesses


if __name__ == '__main__':
    # ---- PART 1.1 -----
    # print(sigmoid(0.35999999999999993))
    # print(d_sigmoid(0.2))

    # --- PART 1.2 ----
    # print(perceptron(np.array([1.0, 2.3, 1.9]),np.array([0.2,0.3,0.1])))
    # print(perceptron(np.array([0.2,0.4]),np.array([0.1,0.4])))

    # ---- PART 1.3 -----
    """np.random.seed(1234)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    
    # Take one point:
    # x = train_features[0, :]
    x = [6.3, 2.5, 4.9, 1.5]
    # print(x)
    K = 3 # number of classes
    M = 10
    D = 4
    # Initialize two random weight matrices
    # np.random.seed(1234)
    W1 = 2 * np.random.rand(D + 1, M) - 1
    # print(W1)
    # np.random.seed(1234)
    W2 = 2 * np.random.rand(M + 1, K) - 1
    # print(W2)
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)

    # ---- PART 1.4 -----
    # np.random.seed(42)
    K = 3  # number of classes
    M = 6
    D = train_features.shape[1]
    x = features[0, :]
    # x = [6.3, 2.5, 4.9, 1.5]

    # create one-hot target for the feature
    target_y = np.zeros(K)
    # setting first value to 1 target_y[0] = 1 (yellow)
    target_y[targets[0]] = 1.0

    # Initialize two random weight matrices
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    # print(W1)
    # print(W2)
    y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)"""

    # ---- PART 2.1 -----
    """
    # initialize the random seed to get predictable results
    np.random.seed(23)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    x = train_features[0, :]
    K = 3
    M = 6
    D = train_features.shape[1]
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    y, z0, z1, a1, a2 = ffnn(x, M, K, W1, W2)
    # target_y = np.zeros(K)
    # target_y[targets[0]] = 1.0
    # y, dE1, dE2 = backprop(x, target_y, M, K, W1, W2)
    # print(y)
    # print(dE1)
    # print(dE2)
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    print(f'W1tr: {W1tr}')
    print(f'W2tr: {W2tr}')
    print(f'Etotal: {Etotal}')
    print(f'misclassification_rate: {misclassification_rate}')
    print(f'last_guesses: {last_guesses}')
    """

    # ---- PART 2.2 -----
    """
    np.random.seed(23)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    K = 3
    M = 6
    D = train_features.shape[1]
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features[:20, :], train_targets[:20], M, K, W1, W2, 500, 0.1)
    X_test = test_features
    # print(len(test_features))
    guesses = test_nn(X_test, M, K, W1tr, W2tr)
    """

    # --- PART 2.3 ---- training the network
    np.random.seed(23)
    features, targets, classes = load_iris()
    (train_features, train_targets), (test_features, test_targets) = split_train_test(features, targets)
    K = 3
    M = 6
    D = train_features.shape[1]
    W1 = 2 * np.random.rand(D + 1, M) - 1
    W2 = 2 * np.random.rand(M + 1, K) - 1
    iterations = 5000
    W1tr, W2tr, Etotal, misclassification_rate, last_guesses = train_nn(train_features, train_targets, M, K, W1, W2, iterations, 0.1)
    guesses = test_nn(test_features, M, K, W1tr, W2tr)
    print(misclassification_rate[-1])
    """
    # accuracy
    accuracy = np.sum(guesses == test_targets) / len(test_targets)
    print(f"The accuracy is: {accuracy}")
    # Confusion Matrix
    confusion_mat = confusion_matrix(test_targets, guesses)
    print(f"The confusion matrix {confusion_mat}")
    # plotting E_total as function of iterations
    plt.plot(range(0, iterations), Etotal)
    plt.title("E_total as a function of iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Etotal")
    plt.show()
    # plotting the misclassification rate as a function of iterations
    plt.plot(range(0, iterations), misclassification_rate)
    plt.title("Misclassification rate as a function of iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Misclassification rate")
    plt.show()
    """

