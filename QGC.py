from scipy.sparse import spdiags
import logging.config
from scipy.sparse import csr_matrix
import numpy as np
from spIdentityMinus import del_sp_row_col, del_sp_row_col2, sp_insert_rows
from scipy.sparse.linalg import eigsh, LinearOperator
from VectorClustering import VectorClustering
from QOCut import QOCut
import time

log = logging.getLogger('test.QGC')
np.set_printoptions(threshold=100000, linewidth=1000)


def QGC_batch(matG, maxitr, vecRel, N, query, K, H, M):
    rank_type = 'eigXrel'
    """ QGC without clustering balance constraint """
    ok = False
    times = 0
    len_m = len(M)
    while not ok and times < 5:
        times += 1
        (weight_eigvec, vecQ) = QGC(matG, maxitr, query, K, 3, vecRel, 1, -0.02, 100)

        if len_m > 1:
            print('還沒做完1')
            NCut_BestRel = np.zeros([len_m, 1])
            vecBestPerform = np.zeros([N, K, len_m])
            vecPerform1 = np.zeros([N, K, len_m])
            for i in M:
                (A_NCut_BestRel, A_vecBestPerform, A_vecPerform1) = QOCut(query, weight_eigvec, vecQ, 'ncut', vecRel,
                                                                          rank_type, [i, 'TotalN'])
                # NCut_BestRel(i) = A_NCut_BestRel
                # vecBestPerform(:,:, i) = A_vecBestPerform
                # vecPerform1(:,:, i) = A_vecPerform1
        elif M[0] > 0:
            print('還沒做完2')
        else:
            # (NCut_BestRel, vecBestPerform, vecPerform1) = QOCut(query, weight_eigvec, vecQ, 'ncut', vecRel, rank_type)
            (NCut_BestRel2, vecBestPerform2, vecPerform2) = QOCut(matG, N, query, weight_eigvec, vecQ, 'ncut', vecRel, rank_type, [])
        vecNcut_rel2 = [NCut_BestRel2]

        if len(vecQ.sum(axis=0).nonzero()[0]) is K:
            ok = True
    NCut_BestRel, vecBestPerform, vecPerform1 = NCut_BestRel2, vecBestPerform2, vecPerform2
    return weight_eigvec, vecQ, vecPerform1, vecBestPerform, NCut_BestRel, vecPerform2, vecBestPerform2, NCut_BestRel2


def QGC(matG, maxitr, query, K, H, vecRel, MyLancType, threshold, eta):
    """
    Function: QOGC_QGC
    This function performs Query-oriented Graph Clustering.
    --------------------------------------------------------------------------------------
    :param matG:
    :param maxitr:
    :param query: queryID
    :param K: the number of clustering. (K = 0 ===> auto-determine the number of clusters K)
    :param H: the top-H eigenvectors are obtained for clustering
    :param vecRel: relevance vector
    :param MyLancType: approach selection index
    :param threshold:
    :param eta: is used only when MyLancType = 2, often set to a huge value

    :return: weight_eigvec :
    :return: vecQ :
    """

    N = matG.shape[0]
    oPhi = 0
    weight_eigvec = 0

    """ Initial setting """
    max_clustering_itr = 200  # 1  # 200

    zeta = 0.05
    q_size = 1

    # 檢查 vecRel需為 1*n 才能進 spdiags()
    if vecRel.shape[0] != 1:
        vecRel = vecRel.reshape(1, vecRel.shape[0])

    matR = spdiags(vecRel[:], 0, N, N)
    """ 轉回來 """
    vecRel = vecRel.reshape(vecRel.shape[1], 1)

    """ 沒用到... """
    # vecD = matG.sum(axis=0)
    # matD = spdiags(vecD[:], 0, N, N)
    """ 沒用到... """

    # vecRD = sum(matR * (matG * matR)).T
    vecRD = (matR * (matG * matR)).sum(axis=0).T

    vec_val = np.asarray(vecRD)
    # vec_val = vecRD.toarray()

    # vecRD_sqrt = [1. / (x[0] ** 0.5) for x in vec_val]
    """ 忍痛改得超複雜.... """
    vecRD_sqrt = list()
    for i in range(vec_val.shape[0]):
        if vec_val[i][0] > 0:
            vecRD_sqrt.append(1. / (vec_val[i][0] ** 0.5))
        else:
            vecRD_sqrt.append(0)
    # print('vecRD_sqrt', vecRD_sqrt)

    while np.inf in vecRD_sqrt:
        idx = vecRD_sqrt.index(np.inf)
        vecRD_sqrt[idx] = 0

    matRD_sqrt = spdiags(vecRD_sqrt[:], 0, N, N)
    # log.debug('matRD_sqrt:\n{0}'.format(matRD_sqrt))

    matG_SC = matR * matG * matR
    matG_SC = matRD_sqrt * matG_SC * matRD_sqrt
    matG_SC = 0.5 * (matG_SC + matG_SC.transpose())

    """ Query setting """
    if query >= 0:
        q_size = 1

        # matG_SC = del_sp_row_col(matG_SC, query)
        matG_SC = del_sp_row_col2(matG_SC, query)

        rel = vecRel.copy()
        # rel[query, :] = 0
        rel = np.delete(rel, query, axis=0)
    else:
        q_size = 0
        rel = vecRel.copy()

    def myGG(x):
        tmp = x - np.dot(rel, (np.dot(rel.T, x)))
        tmp = matG_SC * tmp
        y = tmp - np.dot(rel, np.dot(rel.T, tmp))
        return y

    n = N - q_size
    A = LinearOperator((n, n), matvec=myGG)

    """ Run Graph Clustering """
    clustering_time = 0
    kmeans_run = 1
    while kmeans_run and clustering_time < max_clustering_itr:
        clustering_time += 1
        """ Minimize the objective function by using Langrange multiplier """
        rel /= (rel.transpose().dot(rel)) ** 0.5

        eigVal, eigVec = eigsh(A, maxiter=maxitr, which='LA', k=H)
        eigVal = np.diag(eigVal)

        weight_eigvec = eigVec - np.dot(rel, np.dot(rel.T, eigVec))

        log.info('finish eigenmaps => QGCB 1')

        """ Label assignment  """
        weight_eigvec = np.dot(weight_eigvec, (eigVal + 10e-5))
        (oPhi, K) = VectorClustering(weight_eigvec, K, threshold)

        sizePhi = np.sum(oPhi, axis=0).tolist()[0]
        if 0 not in sizePhi:
            kmeans_run = False

    """ Insert the query into the clustering result """
    if query >= 0:
        mtx_zeros = csr_matrix(np.zeros([1, K]))
        vecQ = sp_insert_rows(oPhi, mtx_zeros, query)
    else:
        vecQ = oPhi
    return weight_eigvec, vecQ


if __name__ == '__main__':
    # import numpy as np
    # vecRel = np.array([[5],
    #           [8],
    #           [9],
    #           [3],
    #           [7]])
    # vecRel = vecRel.reshape(1, 5)
    # # vecRel = [5,8,9,3,7]
    # s = spdiags(vecRel[:], 0, 5, 5)
    # print(s.todense())
    pass
