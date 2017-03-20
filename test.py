from scipy.sparse import coo_matrix, eye
from sp_normalize import SCNomalization
from pagerank import oneQ_pagerank
from QGC import *
import logging
import logging.config
import math
import numpy as np
import time
from LoadGraphData import load_graph_data, load_graph_UIT

""" ---------------- init log config -------------------"""
# log = logging.getLogger('test')
# log.setLevel(logging.DEBUG)
#
# """ file_hdlr = logging.FileHandler('log/QGC_{0}.log'.format(time.time()))"""
# file_hdlr = logging.FileHandler('log/QGC_0.log')
# file_hdlr.setLevel(logging.DEBUG)
#
# console_hdlr = logging.StreamHandler()
# console_hdlr.setLevel(logging.INFO)
#
# formatter = logging.Formatter('%(levelname)-8s - %(asctime)s - %(name)-12s - %(message)s')
# file_hdlr.setFormatter(formatter)
# console_hdlr.setFormatter(formatter)
#
# log.addHandler(file_hdlr)
# log.addHandler(console_hdlr)
#
# log.info('QGC start')
# ------------------- end log config -----------------------


def func_entropy(v):
    """ Calculate the entropy of the given vector v """
    tmp = (v / np.sum(v))
    ret = [math.log(i) for i in tmp.tolist()[0]] * -tmp.T
    return ret.tolist()[0]

alpha = 0.1
t = 10
maxitr = 200

""" Test Type => 1: toy graph; 2: UIT graph; 3: UIT graph; 4: DBLP graph """
test_type = 2

""" Re-read => 0: not re-read; 1: re-read """
# reread = 1

K = 3
topN = [0]

""" algorithm active """
act_QOGC_Tree = 0
act_QOGC_L2Norm = 0
act_QOGC_KLD = 0
act_QOGC_QGC = 1
act_QOGC_QGC_topAll = 0
act_QOGC_AC = 0
act_QOGC_SC = 0

queries = list()
""" load data matG """
if test_type is 1:
    (matG, n, e) = load_graph_data('TestData/simpleGraph_new.txt')
elif test_type is 2:
    (matG, n, e) = load_graph_UIT('TestData/tags.dat', 'TestData/user_taggedartists.dat', 't', 5, 10)


st = time.time()
""" Create symmetric normalized matrix A of G """
matA, matB = SCNomalization(matG)

matI = eye(n, n)
xi = 0.2
matA = (1-xi) * matA + xi * matI


""" Test """
if test_type is 1:
    queries = [0]
elif test_type is 2:
    queries = [0, 2, 3, 4, 5]
    # queries = [0, 2, 3, 4, 5, 18, 20, 22, 31, 38, 40, 55, 61, 66, 74, 77, 78, 79, 80, 81, 82, 85, 86]

""" query | Tree_NCut | L2Norm_NCut | KLD_NCut | Tree_topN | L2Norm_topN | KLD_topN """
# cellResult = cell(length(queries), 19);

time_query = list()
""" Set the queries and their relevance """
for query in queries:
    time_test = time.time()

    if query < 0:
        continue

    """ if the query is isolated, ignore it. """
    # if q >= 0 and sum(matA[q, :]) == 0.2:
    #     continue

    """ Get the relevance vector """
    t_pagerank = time.time()
    if query >= 0:
        vecRel = oneQ_pagerank(matB, query, 0.5)
        vecRel[query, 0] = 0
    else:
        vecRel = np.ones((n, 0)) / n
    print('pagerank: {0} s'.format(time.time() - t_pagerank))

    if vecRel.sum() is 0:
        continue
    # log.info('shape of vecRel: {0}'.format(vecRel.shape))
    """ 確定與 matlab 結果一樣～ """
    # for i in vecRel:
    #     print(np.round(float(i), 8))

    """ Algorithm: QGC5_QGC(Query-oriented spectral clustering)"""
    if act_QOGC_QGC is 1:
        log.info('Run Query-oriented spectral clustering algorithm...')
        # QGC(matG, maxitr, query, K, 3, vecRel, 0, -0.02, 100)
        (iPhi, vecQ, vecPerform1, QGC_vecBestPerform, QGC_NCut_BestRel, vecPerform2, QGC_vecBestPerform2, QGC_NCut_BestRel2) = QGC_batch(matG, maxitr, vecRel, n, query, K, 3, topN)

        c_size1 = vecPerform1.sum(axis=0)
        c_size2 = vecPerform2.sum(axis=0)
        entropy_QGC1 = func_entropy(c_size1)
        print('c_size1:', c_size1)
        print('entropy_QGC1:', entropy_QGC1)

        # entropy_QGC1 = funcEntropy(c_size1')
        # entropy_QGC2 = funcEntropy(c_size2')

    #     [iPhi, vecQ, vecPerform1, QGC_vecBestPerform, QGC_NCut_BestRel, vecPerform2, QGC_vecBestPerform2,
    #      QGC_NCut_BestRel2, vecPerform3, QGC_vecBestPerform3, QGC_NCut_BestRel3] = QOGC_QGC_batch(query, K, 3, topN)
    #     c_size1 = sum(full(vecPerform1))
    #     c_size2 = sum(full(vecPerform2))
    #     c_size3 = sum(full(vecPerform3))
    #     entropy_QGC1 = funcEntropy(c_size1');
    #     entropy_QGC2 = funcEntropy(c_size2');
    #     entropy_QGC3 = funcEntropy(c_size3');
    #     else
    #     if act_QOGC_QGC_topAll == 0
    #     QGC_NCut_BestRel = 0;
    #     QGC_vecBestPerform = sparse(N, K);
    #     QGC_NCut_BestRel2 = 0;
    #     QGC_vecBestPerform2 = sparse(N, K);
    #     QGC_NCut_BestRel3 = 0;
    #     QGC_vecBestPerform3 = sparse(N, K);
    #     entropy_QGC1 = 0;
    #     entropy_QGC2 = 0;
    #     entropy_QGC3 = 0;
    time_query.append(time.time() - time_test)
et = time.time()
print('\ntotal process time no load data: {0} s'.format(et - st))
print('\ntotal for {2} querys :{0} s\n'
      'average time for each query: {1} s'.format(np.sum(time_query), np.sum(time_query)/len(time_query), len(time_query)))
