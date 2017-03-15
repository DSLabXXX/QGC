import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, find
from spIdentityMinus import sp_insert_rows
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
    NCut_rel = 0
    vecRankScore = 0
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

    """
    做不出點乘 用了一個蠢方法 eye * values (func 2)
    直接multiply 轉成dense 後再轉回sparse (func 1)
    func 1 矩陣小時較快
    func 2 矩陣大到一定程度 ex (50000, 300) 才會快過func 1
    """
    # print('vecLabel:\n', vecLabel.todense())

    # func 2
    # d = lil_matrix((N, N))
    # d.setdiag(vecRel)
    # vecPerform = d * vecLabel

    # func 1 (改進 + csr_matrix)
    vecPerform = vecLabel.multiply(csr_matrix(vecRel))
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

        """ sparse ver. for vecPerform_sum """
        # vecPerform_sum = csr_matrix(np.ones((1, N))) * csr_matrix(np.tile(vecRel, [1, K])).multiply(matG * vecPerform)

        """ dense(ndarray) ver. for vecPerform_sum """
        vecPerform_sum = np.dot(np.ones((1, N)), np.multiply(np.tile(vecRel, [1, K]), (matG * vecPerform).A))
        # print('vecPerform_sumvecPerform_sumvecPerform_sum:\n', vecPerform_sum)

        # Normalization
        # vecPerform_norm = vecPerform ./ np.tile(vecPerform_sum, [N, 1])
        vecPerform_norm = vecPerform / np.tile(vecPerform_sum, [N, 1])
        # print('np.tile(vecPerform_sum, [N, 1]:\n', vecPerform_norm)

        """ Calculate Cut """
        vecTemp = np.tile(vecRel, [1, K]) * (matG * vecPerform_norm).A - 10e5 * vecPerform_norm
        vecTemp[vecTemp < 0] = 0
        # print('vecTemp:\n', vecTemp)
        NCut_rel = np.sum(np.sum(vecTemp))
        # print('NCut_rel:', NCut_rel)
    # end if

    """ Ranking for each cluster """
    # eigIndex = insertrows(weight_eigvec, zeros(1, size(weight_eigvec, 2)), query - 1)
    eigIndex = sp_insert_rows(weight_eigvec, np.zeros((1, weight_eigvec.shape[1])), query).tocsr()

    if rank_type == 'rel':
        pass
    elif rank_type == 'eig':
        pass
    elif rank_type == 'eigXrel':
        """ according to the (eigenvector .* releance) """
        vecRankScore = lil_matrix(np.zeros((N, K)))

        for i in range(K):
            [x, y, z] = find(vecQ[:, i])

            if len(x) > 1:
                mean_vec = eigIndex[x, :].mean(axis=0)
            else:
                mean_vec = eigIndex[x, :]
            # print('mean_vec:', mean_vec)
            tmp = eigIndex * mean_vec.T
            # print('tmp:\n', tmp)
            vecRankScore[:, i] = vecQ[:, i].multiply(tmp)
        vecRankScore = vecRankScore.multiply(csr_matrix(vecRel))
        # print('vecRankScore:\n', vecRankScore.todense())
        # vecRankScore = bsxfun( @ times, vecRankScore, vecRel);

    elif rank_type == 'eigXrel2':
        pass
    elif rank_type == 'eig+rel':
        pass

    return NCut_rel, vecRankScore, vecPerform
