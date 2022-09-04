# Author: 
# Date:
# Project: 
# Acknowledgements: 
#

from tools import scatter_3d_data, bar_per_axis

from turtle import update
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# DOES NOT IMPORT FROM TOOLLS

# def scatter_3d_data(data: np.ndarray):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data[:, 0], data[:, 1], data[:, 2])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()


# def bar_per_axis(data: np.ndarray):
#     for i in range(data.shape[1]):
#         plt.subplot(1, data.shape[1], i+1)
#         plt.hist(data[:, i], 100)
#         plt.title(f'Dimension {i+1}')
#     plt.show()

# GRRR



def gen_data(
    n: int,
    k: int,
    mean: np.ndarray,
    var: float
) -> np.ndarray:
    '''Generate n values samples from the k-variate
    normal distribution
    '''
    eye_matrix = np.identity(k)
    var_matrix = var**2 * eye_matrix
    data = np.random.multivariate_normal(mean, var_matrix, n)
    return data


def update_sequence_mean(
    mu: np.ndarray,
    x: np.ndarray,
    n: int
) -> np.ndarray:
    '''Performs the mean sequence estimation update
    '''
    update_mean = mu + 1/ n * (x - mu)
    return update_mean

def _plot_sequence_estimate(mu: np.ndarray):
    # data = gen_data(100, 3, np.array([0, 0, 0]), np.sqrt(3))
    data = gen_data(100, 3, mu, np.sqrt(3))
    estimates = [np.array([0, 0, 0])]

    for i in range(data.shape[0]):
        new_estimate = update_sequence_mean(estimates[-1],data[i], i+1)
        estimates.append(new_estimate)
    
    plt.plot([e[0] for e in estimates], label='First dimension')
    plt.plot([e[1] for e in estimates], label='Second dimension')
    plt.plot([e[2] for e in estimates], label='Third dimension')
    
    plt.legend(loc='upper center')
    plt.show()
    return estimates


def _square_error(y, y_hat):
    # y_hat 303*3 matrix
    # y = 1*3 list
    sqr_error = []
    for i in range(len(y_hat)):
        foo = y - y_hat[i]
        error = np.power(foo, 2)
        mu = np.mean(error)
        sqr_error.append(mu)
    return sqr_error



def _plot_mean_square_error():

    sq_er = _square_error(actual_mean, est_mean)
    x_axiz = range(1,len(sq_er))
    # from second point in data whereas starting points is 0
    plt.plot(x_axiz, sq_er[1::])
    plt.show()


# Naive solution to the independent question.

def gen_changing_data(
    n: int,
    k: int,
    start_mean: np.ndarray,
    end_mean: np.ndarray,
    var: float
) -> np.ndarray:
    # 500 ticks
    mean_updating = np.linspace(start_mean, end_mean, 500)

    updated_estimates = [np.array([0, 0, 0])]
    forget = np.array([0,0,0])
    final_data = []
    counter = 0

    for k in range(len(mean_updating)):
        updated_data = gen_data(500, 3, mean_updating[k], np.sqrt(3))
        final_data.append(updated_data[k])
        counter += 1

    for i in range(len(final_data)):
        new_estimate = update_sequence_mean(updated_estimates[-1],final_data[i], i+1)
        updated_estimates.append(new_estimate)

        # best results to run without forgetting...
        # KIJKA BETUR A ÞETTA

        # if i == 100:
        #     print(new_estimate)
        #     updated_estimates.append(forget)
        # if i == 400:
        #     updated_estimates.append(forget)
        # if i == 300:
        #     updated_estimates.append(forget)
        # if i == 400:
        #     updated_estimates.append(forget)
        # if i == 250:
        #     updated_estimates.append(forget)
        # if i == 350:
        #     updated_estimates.append(forget)
        # Forget data 100 tick fresti bætir ? nýr núll punktur...
    
    plt.plot([e[0] for e in updated_estimates], label='First dimension')
    plt.plot([e[1] for e in updated_estimates], label='Second dimension')
    plt.plot([e[2] for e in updated_estimates], label='Third dimension')
    
    plt.legend(loc='upper center')
    plt.show()
    return updated_estimates


def _plot_changing_sequence_estimate():
    # remove this if you don't go for the independent section
    ...


if __name__ == '__main__':
    # PART 1.1
    np.random.seed(1234)
    # print(gen_data(2,3, np.array([0, 1, -1]), 1.3))
    #print(gen_data(5, 1, np.array([0.5]), 0.5))

    # PART 1.2
    # create X with gen_data
    X = gen_data(300, 3, np.array([0, 1, -1]), np.sqrt(3))
    # scatter_3d_data(X)
    # bar_per_axis(X)

    # PART 1.4
    mean = np.mean(X, 0)
    # print(mean)
    new_x = gen_data(1, 3, np.array([0, 0, 0]), 1)
    # print(new_x)
    # print(X.shape[0])
    #print(update_sequence_mean(mean, new_x, X.shape[0]))

    # PART 1.5
    # I changed so that the plot takes in new mean...
    # did not have any input
    _plot_sequence_estimate(np.array([1,-1,0]))

    # PART 1.6
    ## CALCULATIONS for square error??
    actual_mean = np.array([0,0,0])
    ## Not getting same plot as in instructions.
    est_mean = _plot_sequence_estimate(actual_mean)
    _square_error(actual_mean, est_mean)
    _plot_mean_square_error()

    # INDEPENDENT PART
    gen_changing_data(500, 3, np.array([0,1,-1]), np.array([1,-1,0]), np.sqrt(3))


    




