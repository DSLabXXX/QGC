from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from spIdentityMinus import csr_zero_rows, csr_row_set_nz_to_val
import time
import math


# np.set_printoptions(threshold=100000, linewidth=1000)


def load_graph_data(fp):
    """ fp:file path """
    list_u = list()
    list_i = list()
    list_r = list()
    with open(fp) as f:
        for line in f:
            ls = line.split()
            list_u.append(float(ls[0]) - 1)
            list_i.append(float(ls[1]) - 1)
            list_r.append(1.)
    # log.info('list_u : {0}'.format(list_u))
    # log.info('list_i : {0}'.format(list_i))

    n = int(max(max(list_u), max(list_i)) + 1)
    e = len(list_u)

    matG = coo_matrix((list_r, (list_u, list_i)), shape=(n, n)).tocsr()
    matG = matG + matG.transpose()
    return matG, n, e


def load_graph_UIT(vetex_filepath, UIT_filepath, graphtype, upperbound, lowerBound):
    """
    % --- for tag recommendation ---
    % tag.dat: id \t tag
    % vetex_filepath = 'tags.dat'
    % UIT_filepath = 'user_taggedartists.dat'
    % upperbound = 3
    % lowerBound = 0
    %
    % --- for artist recommendation ---
    % tag.dat: id \t artist \t lastfm_web_url \t pic_url
    % vetex_filepath = 'artists2.txt'
    % UIT_filepath = 'user_taggedartists.dat'
    % upperbound = 1
    """
    """ Read file and Construct matrices """
    list_u = list()
    list_i = list()
    list_t = list()
    setTagID = set()
    i = 0
    with open(vetex_filepath) as f:
        for line in f:
            ls = line.split()
            setTagID.add(int(ls[0])-1)

    with open(UIT_filepath) as f:
        for line in f:
            # if i > 5:
            #     break
            i += 1
            ls = line.split()
            valRowIndex = int(ls[1]) - 1
            valcolIndex = int(ls[2]) - 1
            if valcolIndex in setTagID:
                list_i.append(valRowIndex)
                list_t.append(valcolIndex)
    # ni = max(list_i) + 1
    # nt = max(list_t) + 1
    matIT = coo_matrix((np.ones(len(list_t)), (list_i, list_t))).tocsr()
    print(matIT.shape)

    """ for testing """
    # matIT = csr_matrix([[6, 0, 0, 0, 17], [4, 5, 0, 6, 4], [0, 2, 3, 1, 3],
    #                     [1, 0, 1, 0, 0], [0, 8, 8, 0, 1]])
    # print(matIT.todense())
    # print(matIT.nnz)

    if graphtype == 'i':
        """ 還沒做好.... """
        """ Prune unpopular tags """
        # tagFreq = sum(matIT)
        #
        # tooSmall = tagFreq < lowerBound
        #
        # tagFreq[:, tooSmall.nonzero()[1]] = 0
        # csr_zero_rows(matIT, tooSmall.nonzero()[1])
        #
        # """ Prune too popular items """
        # threshold = np.mean(tagFreq) + upperbound * np.std(tagFreq)
        # for idx in range(len(tagFreq)):
        #     if tagFreq[0, idx] < threshold:
        #         csr_row_set_nz_to_val(matIT, idx)
    elif graphtype == 't':
        """ Prune unpopular items """
        itemFreq = sum(matIT.T).A

        tooSmall = itemFreq < lowerBound

        itemFreq[:, tooSmall.nonzero()[1]] = 0
        csr_zero_rows(matIT, tooSmall.nonzero()[1])

        """ Prune too popular items """
        threshold = np.mean(itemFreq) + upperbound * np.std(itemFreq)
        list_to_rm = list()
        for idx in range(itemFreq.shape[1]):
            if itemFreq[0, idx] > threshold:
                list_to_rm.append(idx)
        csr_zero_rows(matIT, list_to_rm)

    """ reweight the influence of a tag / item based on TFIDF. """
    matLargeThan1 = matIT > 0
    print('matIT.nnz:\n', matIT.nnz)
    print('matLargeThan1.nnz:\n', matLargeThan1.nnz)

    if graphtype == 't':
        tmp = np.sum(matLargeThan1, axis=1)
        tmp = tmp.reshape(tmp.shape[0]).tolist()[0]
        vecItem_pop = np.array([math.log2(i+2) for i in tmp])
        """ 除完會變成ndarray.matrix FK """
        matIT /= vecItem_pop.reshape(vecItem_pop.shape[0], 1)
        matIT = csr_matrix(matIT)
        # print('matIT.shape:\n{0}\nmatIT /= vecItem_pop:\n{1}'.format(matIT.shape, matIT))

        """ 03/08 寫到這 QQ """
        matrixVertex = matIT.T * matIT



if __name__ == '__main__':
    load_graph_UIT('/home/c11tch/workspace/PycharmProjects/QGC/TestData/tags.dat',
                   '/home/c11tch/workspace/PycharmProjects/QGC/TestData/user_taggedartists.dat',
                   't', 5, 10)
