import numpy as np
import matplotlib.pyplot as plt


def normal(x: np.ndarray, sigma: float, mu: float) -> np.ndarray:
    # Part 1.1
    variable = (x - mu)
    fun1 = 1/(np.sqrt(2 * np.pi * np.power(sigma, 2)))
    fun2 = np.exp(-((np.power(variable,2)/(2*np.power(sigma,2)))))
    prob_distr = fun1*fun2
    return prob_distr


def plot_normal(sigma: float, mu:float, x_start: float, x_end: float):
    # Part 1.2
    x = np.linspace(x_start, x_end, 200)
    pdf = normal(x, sigma, mu)
    plt.plot(x,pdf)
    #plt.show()

def _plot_three_normals():
    # Part 1.2

# def normal_mixture(x: np.ndarray, sigmas: list, mus: list, weights: list):
#     # Part 2.1

# def _compare_components_and_mixture():
#     # Part 2.2

# def sample_gaussian_mixture(sigmas: list, mus: list, weights: list, n_samples: int = 500):
#     # Part 3.1

# def _plot_mixture_and_samples():
#     # Part 3.2

if __name__ == '__main__':
    # Part 1.1
    # normal(0,1,0)
    # normal(3,1,5)
    # normal(np.array([-1,0,1]),1,0)

    # PART 1.2
    #plot_normal(0.5, 0, -2, 2)

    plot_normal(0.5, 0, -5, 5)
    plot_normal(0.25, 1, -5, 5)
    plot_normal(1,1.5, -5, 5)
    plt.show()




    # select your function to test here and do `python3 template.py`
