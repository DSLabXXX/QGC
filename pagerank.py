import numpy as np

def oneQ_pagerank(mat, q, alpha):
    n = mat.shape[0]
    if q == -1:
        # PageRank
        vecP = np.ones((n, 1)) / n
    else:
        # Personalized PageRank
        vecP = np.zeros((n, 1))
        vecP[q, 0] = 1

    vecR = vecP
    for i in range(50):
        vecR = (1-alpha) * (mat * vecR) + alpha * vecP
    return vecR