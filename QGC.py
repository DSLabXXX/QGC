"""
Function: QOGC_QGC
This function performs Query-oriented Graph Clustering.
--------------------------------------------------------------------------------------
query: queryID
K: the number of clustering. (K = 0 ===> auto-determine the number of clusters K)
H: the top-H eigenvectors are obtained for clustering
vecRel: relevance vector
MyLancType: approach selection index
eta: is used only when MyLancType = 2, often set to a huge value
"""

from scipy.sparse import spdiags
import logging.config
from scipy.sparse import lil_matrix, coo_matrix
import numpy as np
from spIdentityMinus import del_sp_row_col
from scipy.sparse.linalg import eigsh, LinearOperator
from VectorClustering import VectorClustering

log = logging.getLogger('test.QGC')
np.set_printoptions(threshold=100000, linewidth=1000)


def QGC(matG,  maxitr, query, K, H, vecRel, MyLancType, threshold, eta):
    N = matG.shape[0]

    """ Initial setting """
    max_clustering_itr = 1 # 200

    zeta = 0.05
    q_size = 1

    # 檢查 vecRel需為 1*n 才能進 spdiags()
    if vecRel.shape[0] != 1:
        vecRel = vecRel.reshape(1, vecRel.shape[0])

    matR = spdiags(vecRel[:], 0, N, N)

    """ 轉回來 """
    vecRel = vecRel.reshape(vecRel.shape[1], 1)

    # log.debug('matR:\n{0}'.format(matR))
    vecD = sum(matG)
    # log.debug('vecD:\n{0}'.format(vecD))
    # matD = spdiags(vecD[:], 0, N, N)
    # log.debug('matD:\n{0}'.format(matD))
    vecRD = sum(matR * (matG * matR)).T
    log.debug('vecRD:\n{0}'.format(vecRD))

    vec_val = vecRD.toarray()

    # vecRD_sqrt = [1. / (x[0] ** 0.5) for x in vec_val]
    """ 忍痛改得超複雜.... """
    vecRD_sqrt = list()
    for i in range(len(vec_val)):
        if vec_val[i][0] > 0:
            vecRD_sqrt.append(1. / (vec_val[i][0] ** 0.5))
        else:
            vecRD_sqrt.append(0)
    # print('vecRD_sqrt', vecRD_sqrt)

    while np.inf in vecRD_sqrt:
        idx = vecRD_sqrt.index(np.inf)
        vecRD_sqrt[idx] = 0

    matRD_sqrt = spdiags(vecRD_sqrt[:], 0, N, N)
    log.debug('matRD_sqrt:\n{0}'.format(matRD_sqrt))

    matG_SC = matR * matG * matR
    matG_SC = matRD_sqrt * matG_SC * matRD_sqrt
    matG_SC = 0.5 * (matG_SC + matG_SC.transpose())

    """ Query setting """
    if query >= 0:
        q_size = 1

        # print('msc ', matG_SC.todense())
        matG_SC = del_sp_row_col(matG_SC, query)
        # print('msc ', matG_SC.todense())
        rel = vecRel.copy()
        rel[query, :] = 0
        rel = np.delete(rel, query, axis=0)
        # print('query={0}\n rel={1}'.format(query, rel))
    else:
        q_size = 0
        rel = vecRel
    # print(matG_SC)


    def myGG(x):
        # print('x shape', x.shape)
        tmp = x - np.dot(rel, (np.dot(rel.T, x)))
        # print('t shape', tmp.shape)
        tmp = matG_SC * tmp
        # print('t2 shape', tmp.shape)
        y = tmp - np.dot(rel, np.dot(rel.T, tmp))
        # print('y shape', y.shape)
        return y
    # def myGG(v):
    #     return v * 5

    n = N - q_size
    A = LinearOperator((n, n), matvec=myGG)

    # Run Graph Clustering
    clustering_time = 0
    kmeans_run = True
    while kmeans_run and clustering_time < max_clustering_itr:
        clustering_time += 1
        """ Minimize the objective function by using Langrange multiplier """
        rel /= (rel.transpose().dot(rel)) ** 0.5
        log.info('rel.shape: {0}'.format(rel.shape))

        eigVal, eigVec = eigsh(A, maxiter=maxitr, which='LA', k=H)
        eigVal = np.diag(eigVal)

        log.debug('eigVal\n{0}'.format(eigVal))
        log.debug('eigVec\n{0}'.format(eigVec))

        weight_eigvec = eigVec - np.dot(rel, np.dot(rel.T, eigVec))
        log.info('finish eigenmaps => QGCB 1')

        """ ----------------- check line ------------------------------------------------ """

        # """ Label assignment  """
        log.debug('(eigVal + 10e-5)\n{0}'.format(eigVal + 10e-5))
        weight_eigvec = np.dot(weight_eigvec, (eigVal + 10e-5))
        log.info('weight_eigvec\n{0}'.format(weight_eigvec))
        # oPhi, K = VectorClustering(weight_eigvec, K, threshold)
        VectorClustering(weight_eigvec, K, threshold)


if __name__ == '__main__':
    import numpy as np
    vecRel = np.array([[5],
              [8],
              [9],
              [3],
              [7]])
    vecRel = vecRel.reshape(1, 5)
    # vecRel = [5,8,9,3,7]
    s = spdiags(vecRel[:], 0, 5, 5)
    print(s.todense())