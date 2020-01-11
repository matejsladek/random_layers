import numpy as np
import operator
from functools import reduce
def prod(iterable):
    return reduce(operator.mul, iterable, 1)
# from math import prod
from matplotlib import pyplot as plt


dot_size = 3


def plot_marginals(data, dimension_meaning, values, dimension_of_interest):
    dimension_swap = list(range(len(data.shape)))
    del dimension_swap[dimension_of_interest]
    dimension_swap.insert(0, dimension_of_interest)
    data = np.transpose(data, dimension_swap)
    values = np.array(values)[dimension_swap[:-1]]
    dimension_meaning = np.array(dimension_meaning)[dimension_swap[:-1]]

    total_observations = prod(data.shape)

    for i in range(len(data.shape) - 1):
        plt.figure(figsize=(16, 9))
        indices = tuple(filter(lambda index: index != i, range(1, len(data.shape))))
        plt.plot(np.mean(data, axis=indices))

        if i == 0:
            plt.scatter(
                np.arange(data.shape[0]).reshape(-1, 1).repeat(total_observations / data.shape[0], axis=1),
                data.reshape(data.shape[0], -1),
                s=dot_size,
            )
        else:
            dimension_swap = list(range(len(data.shape)))
            del dimension_swap[i]
            dimension_swap.insert(1, i)
            by_relevant_dimension = np.transpose(data, dimension_swap).reshape(data.shape[0], data.shape[i], -1)

            for j in range(data.shape[i]):
                plt.scatter(
                    np.arange(data.shape[0]).reshape(-1, 1).repeat(by_relevant_dimension.shape[-1], axis=1),
                    by_relevant_dimension[:, j, :],
                    s=dot_size,
                )

        plt.title("Varying over " + dimension_meaning[i].lower(), fontsize=22)
        plt.xticks(range(data.shape[0]), values[0])
        plt.legend([str(value) for value in values[i][:i + 1]])
        plt.xlim(-.5, len(values[0]) - .5)
        plt.show()
