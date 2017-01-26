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


def VectorClustering(weight_eigvec, K, threshold):
    p = 10
    topP = 3
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
    log.info('\n{0}'.format(matPartitionLabel))

    """ Initialize the similarity matrix """
    G = matPartitionLabel.shape[1]
    cent_norm = np.zeros((G, 1))
    centroids = np.zeros((G, H))
    variance = np.zeros((G * H, H))
    """ ----------------- check line ------------------------------------------------ """
    for i in range(G):
        # [x, y] = find(matPartitionLabel(:, i))
        x = matPartitionLabel[:, i].nonzero()[0]
        bigk = nlargest(topP, zip(scale_weight_eigvec[x, :], itertools.count()))
        x1 = [i[1] for i in bigk]
        centroid2 = np.mean(weight_eigvec[x[x1], :], axis=0)
        print('weight_eigvec[x[x1], :]\n', weight_eigvec[x[x1], :])
        print('centroid2\n', centroid2)
        cent_norm[i] = np.linalg.norm(centroid2)
        print('cent_norm\n', cent_norm)
        centroid2 /= np.linalg.norm(centroid2)
        centroids[i, :] = centroid2
        # print('centroids', centroids)

        """ mean of the top-P nodes to be centroid  """
        """
            [v1, x1] = maxk(scale_weight_eigvec(x, :), topP);
            centroid2 = mean(weight_eigvec(x(x1,1), :));
            cent_norm(i,1) = norm(centroid2);
            centroid2 = centroid2 / norm(centroid2);
            centroids(i, :) = centroid2;
        """
