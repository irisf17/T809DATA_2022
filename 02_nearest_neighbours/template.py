# Author: Íris Friðriksdóttir
# Date: 8.August 2022
# Project: 02_nearest_neighbours
# Acknowledgements: Bjarki Laxdal, Ignas Sauliusson and Kristján Daðason
# 

import numpy as np
import matplotlib.pyplot as plt
import help
from sklearn.metrics import confusion_matrix
from tools import load_iris, split_train_test, plot_points


def euclidian_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculate the euclidian distance between points x and y
    '''
    sum = 0
    for i in range(len(x)):

        subtract = x[i]-y[i]
        number = np.power(subtract, 2)
        sum = sum + number
    distance = np.sqrt(sum)
    return distance



def euclidian_distances(x: np.ndarray, points: np.ndarray) -> np.ndarray:
    '''
    Calculate the euclidian distance between x and and many
    points
    '''
    distances = []
    for i in range(len(points)):
        dist = euclidian_distance(x, points[i])
        distances.append(dist)

    return distances

def k_nearest(x: np.ndarray, points: np.ndarray, k: int):
    '''
    Given a feature vector, find the indexes that correspond
    to the k-nearest feature vectors in points
    '''
    dist = euclidian_distances(x, points)
    # find index in list that has value closest to x
    sort_arr = np.argsort(dist)
    return sort_arr[0:k]

def vote(targets, classes):
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # out of numbers in classes, count how many times in targets
    #, print out the max number
    max_counter = 0
    counter = 0
    best_value = 0
    for value in classes:
        for target in targets:
            if value == target:
                counter += 1
        if counter > max_counter:
            max_counter = counter
            counter = 0
            best_value = value
    return best_value



def knn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    est_targ = []

    indx = k_nearest(x, points, k)
    for i in indx:
        est_targ.append(point_targets[i])
    #print(est_targ)
    result = vote(est_targ,classes)
    return result
    

def knn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    pred_clas = []
    for i in range(len(points)):
        pred_targ = knn(points[i], help.remove_one(points, i), 
        help.remove_one(point_targets,i), classes, k)
        #print(pred_targ)
        pred_clas.append(pred_targ)
    return pred_clas

def knn_accuracy(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> float:
    N = len(point_targets)
    count = 0
    
    pred_target_list = knn_predict(points, point_targets, classes, k)
    for j in range(len(point_targets)):
        if pred_target_list[j] == point_targets[j]:
            count += 1
    acc = count/N
    return acc
    
    


def knn_confusion_matrix(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:

    pred_targ = knn_predict(points, point_targets, classes, k)
    return confusion_matrix(pred_targ, point_targets)


def best_k(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
) -> int:

    best_acc = 0
    N = len(point_targets)
    for i in range(1,N-1):
        acc = knn_accuracy(points, point_targets, classes, i)
        if acc > best_acc:
            best_acc = acc
            best_k = i
    return best_k


def knn_plot_points(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
):
    # predicted targets
    predict_t = knn_predict(points, point_targets, classes, k)
    colors = ['yellow', 'purple', 'blue']
    for i in range(len(point_targets)):
        [x, y] = points[i,:2]
        if predict_t[i] == point_targets[i]:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='green',
            linewidths=2)
        else:
            plt.scatter(x, y, c=colors[point_targets[i]], edgecolors='red',
            linewidths=2)
    plt.title('Yellow=0, Purple=1, Blue=2')
    plt.show()

def weighted_vote(
    targets: np.ndarray,
    distances: np.ndarray,
    classes: list
) -> int:
    '''
    Given a list of nearest targets, vote for the most
    popular
    '''
    # create an array
    # new_arr = np.zeros((len(classes),len(distances)))
    inverse_arr = []
    weight_arr = []
    c1 = 0
    c2 = 0
    c3 = 0
    class_list = []

    #highest_vote = 1/(distances)
    for i in range(len(distances)):
        if distances[i] == 0:
            inverse_val = 0
        else:
            inverse_val = 1/distances[i]
        #print(f"this is dist: {distances[i]}")
        inverse_arr.append(inverse_val)
    sum_inv = sum(inverse_arr)
    # calculate the weights for each value from distance
    for k in range(len(inverse_arr)):
        weight_val = inverse_arr[k]/sum_inv
        weight_arr.append(weight_val)
    # Each weight is a vote for its associated class
    for j in range(len(weight_arr)):
        if targets[j] == 0:
            c1 += weight_arr[j]
        elif targets[j] == 1:
            c2 += weight_arr[j]
        else:
            c3 += weight_arr[j] 
    class_list.append(c1)
    class_list.append(c2)
    class_list.append(c3)
    best_list = np.argsort(class_list)
    best_indx = best_list[-1]

    return classes[best_indx]

def wknn(
    x: np.ndarray,
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:
    '''
    Combine k_nearest and vote
    '''
    est_clas = []
    nearest_val = k_nearest(x, points, k)
    #indx = k_nearest(x, points, k)
    distances = euclidian_distances(x, points[nearest_val])
    for i in nearest_val:
        est_clas.append(point_targets[i])
    #print(est_targ)
    result = weighted_vote(est_clas, distances, classes)
    return result


def wknn_predict(
    points: np.ndarray,
    point_targets: np.ndarray,
    classes: list,
    k: int
) -> np.ndarray:


    pred_clas_list = []
    for i in range(len(points)):
        pred_t = wknn(points[i], help.remove_one(points, i), help.remove_one(point_targets, i), classes, k)
        #pred_targ = knn(points[i], help.remove_one(points, i), help.remove_one(point_targets,i), classes, k)
        pred_clas_list.append(pred_t)
    return pred_clas_list


def compare_knns(
    points: np.ndarray,
    targets: np.ndarray,
    classes: list
):
    N = len(targets)
    count = 0
    k = list(range(1,28+1))
    knn_acc_list = []
    wknn_acc_list = []

    for k_value in k:
        knn_acc = knn_accuracy(points, targets, classes, k_value)
        knn_acc_list.append(knn_acc)
     # Find accuracy for knn_weighted
    for value in k:
        wknn_pred_t = wknn_predict(points, targets, classes, value)
    
        for i in range(len(targets)):
            if wknn_pred_t[i] == targets[i]:
                count += 1
        wknn_acc = count/N
        wknn_acc_list.append(wknn_acc)
        count=0
    plt.plot(k, knn_acc_list, label = 'knn') # legend
    plt.plot(k, wknn_acc_list, label = 'wknn') # legend
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Part 1.1
    # d, t, classes = load_iris()
    # x, points = d[0,:], d[1:,:]
    # x_target, point_targets = t[0], t[1:]
    # print(euclidian_distance(x, points[0]))
    # print(euclidian_distance(x, points[50]))

    # Part 1.2
    # print(euclidian_distances(x, points))

    # Part 1.3
    ## ---- The np.argsort hint was very helpful! :)
    # print(k_nearest(x, points, 1))
    # print(k_nearest(x, points, 3))

    # Part 1.4
    # print(vote(np.array([0,0,1,2]), np.array([0,1,2])))
    # print(vote(np.array([1,1,1,1]), np.array([0,1])))
    
    # Part 1.5
    ## ---- The value of k should correspond to the value of x_target.
    # print(knn(x, points, point_targets, classes, 1))
    # print(knn(x, points, point_targets, classes, 5))
    # print(knn(x, points, point_targets, classes, 150))

    # PART 2.1
    # d, t, classes = load_iris()
    # (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8)
    # print(knn_predict(d_test, t_test, classes, 10))
    # print(knn_predict(d_test, t_test, classes, 5))

    # PART 2.2
    # print(knn_accuracy(d_test, t_test, classes, 10))
    # print(knn_accuracy(d_test, t_test, classes, 5))

    # PART 2.3
    # print(knn_confusion_matrix(d_test, t_test, classes, 10))
    # print(knn_confusion_matrix(d_test, t_test, classes, 20))

    # PART 2.4
    # print(best_k(d_train, t_train, classes))

    # PART 2.5
    ## ---- Got help from Ignas Sauliusson on how to plot the green and red circles.
    ## ---- It was obvious when he pointed it to me to check to the tools.plot_points
    ## ---- as you mentioned in the instructions.
    # knn_plot_points(d, t, classes, 3)

    # INDEPENDENT PART
    # B.1
    ## ---- Got help from Kristján Daðason to get me started,
    ## ---- I had difficulties with understanding the method.
    ## ---- weighted_vote(targets, distances, classes)
    ## ---- I also used another formula to calculate the weighted vote
    ## ---- found from this link https://visualstudiomagazine.com/articles/2019/04/01/weighted-k-nn-classification.aspx?fbclid=IwAR0nSDdtARs5I2nJQQvx2s_-oKFY-YxPhd7GixS_O0xXfduS4qgxXVBZd7E#:~:text=The%20weighted%20k%2Dnearest%20neighbors,or%20more%20numeric%20predictor%20variables

    # features, targets, classes = load_iris()
    # x, points = features[0,:], features[1:,:]
    # x_target, point_targets = targets[0], targets[1:]
    # nearest_val = k_nearest(x, points, 70)
    # distances = euclidian_distances(x, points[nearest_val])
    # print(weighted_vote(targets[nearest_val], distances, classes))

    # B.2
    # print(wknn(x, points, point_targets, classes, 4))

    # B.3
    # print(wknn_predict(points, point_targets, classes, 4))

    # B.4
    ## ---- Got help from Bjarki Laxdal to remind me to use the test points
    ## ---- in the compare_knns function.
    # d, t, classes = load_iris()
    # (d_train, t_train), (d_test, t_test) = split_train_test(d, t, train_ratio=0.8 )
    # compare_knns(d_test, t_test, classes)

    ''' THEORETICAL PART
    The difference in accuracy between kNN and wKnn when k increases is that
    the accuracy for wKnn improves when there are more points taken into the calculation, 
    but the accuracy for knn decreases when k increases.
 
    The reason for this is that for wKnn, if the distances from the data point increase then
    the weighted vote for those points decreases. So they have less effect on the prediction and calculations.
    But for the knn all the points have equal votes. So the further away the point is the more likelier for a misclassification for the point.
    ''' 