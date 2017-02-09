"""
    % Function: VectorClustering
    % This function is to cluster obtained eigenvectors in the eigenspace.
    % Notice that the similarity is defined as Cosine Similarity.
    % The scale of a vector represents its 'strength'.
    %--------------------------------------------------------------------------------------
    % weight_eigvec: weighted eigenvectors
    % K: the number of clustering. (K = 0 ===> auto-determine the number of clusters K)
"""
import numpy as np
import logging.config
from heapq import nlargest
import itertools

log = logging.getLogger('test.QGC.VC')
np.set_printoptions(threshold=100000, linewidth=1000)


def MaxOfMatrix(X):
    """ Finding MAX value and indices of MATRIX. """
    """
    [[-10.0000  , -0.2772 ,  -0.2962  , -0.6079],
    [ -0.2772 , -10.0000  , -0.1342 ,  -0.2789],
    [-0.2962  , -0.1342 , -10.0000  , -0.3121],
    [-0.6079  , -0.2789,   -0.3121  ,-10.0000]]
    """
    (h, w) = X.shape
    index = np.argmax(X)
    row = index // w
    col = index % w
    return np.max(X), row, col


def VectorClustering(weight_eigvec, K, threshold):
    p = 10
    topP = 3
    centroid_types = ['mean', 'power mean', 'top-P', 'ndist']
    cent_type = centroid_types[2]

    H = weight_eigvec.shape[1]

    scale_weight_eigvec = np.array([np.dot(i, i.T) for i in weight_eigvec]).reshape(weight_eigvec.shape[0], 1)
    # log.info(scale_weight_eigvec)
    log.info('scale_weight_eigvec.shape: {0}'.format(scale_weight_eigvec.shape))

    """ Initialize the clustering tree leaves """
    oPhi = weight_eigvec.copy()
    matPartitionLabel = list()
    # oPhi = np.array([[0,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])
    # a = np.sum((oPhi != 0).sum(0))
    # print(a)
    while np.sum((oPhi != 0).sum(0)) > 0:
        # find first index of non zero value
        i = np.sum(oPhi, axis=1).nonzero()[0][0]
        # print('weight_eigvec', weight_eigvec)
        # print('weight_eigvec[i, :]', weight_eigvec[i, :])
        matMatch = weight_eigvec * weight_eigvec[i, :]
        # print('matMatch1\n', matMatch)
        matMatch = matMatch > 0
        # print('matMatch2\n', matMatch)
        vecMatch = np.sum(matMatch, axis=1)
        # print('vecMatch\n', vecMatch)
        vecMatch = vecMatch >= H

        # print('vecMatch\n', vecMatch.astype(int))
        matPartitionLabel += [vecMatch.astype(int)]

        vecZ = vecMatch == 0
        # print('oPhi\n', oPhi)
        # print('vecZ\n', vecZ.astype(int))
        oPhi *= vecZ.reshape(vecZ.shape[0], 1).astype(int)
        # print('oPhi2\n', oPhi)
    matPartitionLabel = np.array(matPartitionLabel).T
    log.info('matPartitionLabel =\n{0}'.format(matPartitionLabel))

    """ Initialize the similarity matrix """
    G = matPartitionLabel.shape[1]
    cent_norm = np.zeros((G, 1))
    centroids = np.zeros((G, H))
    variance = np.zeros((G * H, H))

    """ ↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 有一排數通常都會不一樣 其他皆與matlab寫出來相同 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓ """
    """ mean of the top-P nodes to be centroid  """
    for i in range(G):
        x = matPartitionLabel[:, i].nonzero()[0]
        bigk = nlargest(topP, zip(scale_weight_eigvec[x, :], itertools.count()))
        x1 = [i[1] for i in bigk]
        we = weight_eigvec[x[x1], :]
        if we.shape[0] > 1:
            centroid2 = np.mean(we, axis=0)
        else:
            centroid2 = np.mean(we, axis=1)
        cent_norm[i] = np.linalg.norm(centroid2)

        centroid2 /= np.linalg.norm(centroid2)
        centroids[i, :] = centroid2
    log.info('cent_norm\n{0}'.format(cent_norm))
    log.info('centroids\n{0}'.format(centroids))
    """ ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 有一排數通常都會不一樣 其他皆與matlab寫出來相同 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ """
    """ ----------------- check line ------------------------------------------------ """
    G = centroids.shape[0]
    matClusterSim = np.zeros((G, G))

    """ non-model prior """
    matClusterSim = np.dot(centroids, centroids.T)
    scale_cluster = np.sqrt(np.diag(matClusterSim))
    matClusterSim = matClusterSim / scale_cluster
    matClusterSim = matClusterSim / scale_cluster

    matClusterSim -= np.eye(G)
    # print(np.eye(G))
    # print('matClusterSim5\n{0}'.format(matClusterSim))
    X = len(matClusterSim)

    """
        Hierarchical agglomerative clustering greedily
           - Facebook data: matClusterSim >= -0.02
           - Twitter data: matClusterSim >= 0.9
    """

    s = (matClusterSim >= threshold)
    n = np.sum(s)

    while (X > K > 0) or (np.sum(matClusterSim >= threshold) > 0 and K == 0):
    # for i in range(1):
        # Choose the most similar eigen vectors and merge them
        (max_v, max_x, max_y) = MaxOfMatrix(matClusterSim)
        vecMerge = matPartitionLabel[:, max_x] + matPartitionLabel[:, max_y]

        """
            delete column index max_x and max_y from  matPartitionLabel.
            merge them and put in final column.
        """
        # print('matPartitionLabel before \n', matPartitionLabel)
        matPartitionLabel = np.delete(matPartitionLabel, [max_x, max_y], 1)
        vecMerge = vecMerge.reshape(vecMerge.shape[0], 1)
        matPartitionLabel = np.hstack(([matPartitionLabel, vecMerge]))
        # print('matPartitionLabelmatPartitionLabelmatPartitionLabel:\n', matPartitionLabel)

        """ Update the similarity matrix """
        G = matPartitionLabel.shape[1]
        centroids = np.zeros((G, H))
        cent_norm = np.zeros((G, 1))
        for i in range(G):
            x = matPartitionLabel[:, i].nonzero()[0]
            y = matPartitionLabel[:, i][matPartitionLabel[:, i].nonzero()]  # 疑似用不到

            if cent_type == 'top-P':
                """ mean of the top-P nodes to be centroid """
                # print('scale_weight_eigvec[x]:\n', scale_weight_eigvec[x].T[0])
                tmp_eigvec = scale_weight_eigvec[x].T[0]
                if tmp_eigvec.shape[0] >= topP:
                    x1 = np.argpartition(tmp_eigvec, -topP)[-topP:]
                else:
                    x1 = np.argpartition(tmp_eigvec, -tmp_eigvec.shape[0])[-tmp_eigvec.shape[0]:]

                """ 0207 HERE """
                centroid2 = np.mean(weight_eigvec[x[x1]], axis=0)
                # print('weight_eigvec[x[x1]]:\n', weight_eigvec[x[x1]], '\ncentroid2: ', centroid2, )
                cent_norm[i] = np.linalg.norm(centroid2)
                # print(cent_norm)
                centroid2 /= np.linalg.norm(centroid2)
                # print('centroid2:\n', centroid2)
                centroids[i] = centroid2
                # print('centroids:\n', centroids)
        # end for
        matClusterSim = np.zeros(G)
        if cent_type == 'ndist':
            pass
        else:
            """ non - model prior """
            matClusterSim = np.dot(centroids, centroids.T)
            scale_cluster = np.sqrt(np.diag(matClusterSim))
            print('matClusterSim:\n', matClusterSim)
            print('scale_cluster.T:\n', scale_cluster.T)
            """ [:,None] 為了做./ matClusterSim = bsxfun(@rdivide, matClusterSim, scale_cluster) """
            matClusterSim /= scale_cluster.T[:, None]
            matClusterSim /= scale_cluster.T
            matClusterSim -= 10 * np.eye(G)
        # end if
        matClusterSim -= np.eye(G)
        X = len(matClusterSim)
    # end while

