import numpy as np
from scipy.sparse import coo_matrix, lil_matrix, eye, csr_matrix
import time


def QOCut(matG, N, query, weight_eigvec, vecQ, func_type, vecRel, rank_type, list_m):
    """
    Function: QOGC_QGC
    This function performs Query-oriented Graph Clustering.
    --------------------------------------------------------------------------------------
    :param query: queryID
    :param weight_eigvec: weighted eigenvectors
    :param vecQ: label assignment
    :param ftype: 'ratiocut' or 'ncut'
    :param vecRel: relevance vector
    :param rank_type: choose a method to rank data for each cluster ['rel', 'eig', 'eigXrel', 'eigXrel2', 'eig+rel']
    :param list_m : A list content :param M and :param m_type
    :param M: QONCut@M => only consider top-M items for each cluster
    :param m_type: 'TopNperCluster' or 'TotalN'

    :return: NCut_BestRel :
    :return: vecBestPerform :
    :return: vecPerform1 :
    """
    K = vecQ.shape[1]
    if len(list_m) > 0:
        M, m_type = list_m[0], list_m[1]

    """ Label assigning """
    # print('vecQvecQvecQvecQ:\n', vecQ.todense())
    # """ 此方法不好 要找時間處理.... """
    # vecIndex = np.argmax(vecQ.A, axis=1)
    # # print(vecIndex)
    # vecLabel = coo_matrix((np.ones(N), (np.arange(N), vecIndex)), shape=(N, K)).tocsr()
    vecLabel = vecQ.copy()

    """ Ranking score is according to the relevance """

    """ 做不出點乘 用了一個蠢方法 eye * values 感覺就超慢 FK... """
    print('vecLabel:\n', vecLabel.todense())

    # st = time.time()
    d = lil_matrix((N, N))
    d.setdiag(vecRel)
    vecPerform = d * vecLabel
    # print('eye * values: ', time.time() - st)
    print('vecPerform:\n', vecPerform.todense())
    """ but 另一個更慢 fk... """
    # st = time.time()
    # vecPerform = csr_matrix(vecLabel.multiply(vecRel))
    # print('multiply: ', time.time() - st)
    # print('vecPerform:\n', vecPerform.todense())
    """ ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 02 / 16 check line ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ """

    if len(list_m) > 0:
        vecLabel = np.zeros((N, K))
        vecRel2 = np.zeros((N, 1))

        if m_type == 'TopNperCluster':
            pass
        elif m_type == 'TotalN':
            pass

        vecPerform = vecLabel
        """ Only consider the top M vertices, left others with relevance 0s. """
        vecRel = vecRel2

    """ Calculate the normalization term """
    if func_type == 'ratiocut':
        pass
    else:
        """ the idea is the same with NCut """
        # vecPerform_sum = ones(1, N) * (repmat(vecRel, [1, K]). * (matG * vecPerform))
        # vecPerform_sum = np.ones((1, N)) * (repmat(vecRel, [1, K]).* (matG * vecPerform))

