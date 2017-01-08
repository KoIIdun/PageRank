from bak_ost import bak_ost
from power_m import power_m
import math
import numpy as np
import time

def clog(pagerank):
    vector = list(sorted(pagerank, reverse=True))
    k = [math.log2(i) for i in range(1, len(vector) + 1)]
    y = [math.log2(i) for i in vector]
    A = np.vstack([k, np.ones(len(k))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return m

def main():
    #a_variants - массив тестируемых притягательностей вершины
    #result - файл вывода данных
    #pagerank - вектор PageRank для текущего генерируемого случая
    result = open("test_res.txt", "w")
    a_variants = [30, 40, 60, 70]
    for a in a_variants:
        result.write(str(a) + "\n")
        for k in range(1):
            graph = bak_ost(1000000, 10, a)
            t = time.time()
            pagerank = power_m(graph)
            t  = time.time() - t
            lg = clog(pagerank)
            result.write(str(lg) + " " + str(t) + "\n")
    result.close()

main()