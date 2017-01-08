#Power method

import numpy as np

def read(graph):
    n = graph.shape[0]
    for i in range(n):
        graph[i, i] -= 1
    graph = graph.transpose()
    return graph.tocoo().tocsr(), n

def f(A, x):
    return 0.5 * np.linalg.norm(A.dot(x)) ** 2

def power_m(graph):
    #graph - сгенерированный граф для которого нужно посчитать вектор PageRank
    #x - вектор PageRank для graph
    #EPS - точность с вычисления
    A, n = read(graph)
    x = np.array([0.0] * n)
    x[0] = 1.0
    EPS = 10 ** (-7)
    while f(A, x) > EPS:
        x = A.dot(x) + x
    x = x / x.sum()
    print(*x)
    return x
