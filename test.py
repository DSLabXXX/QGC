import numpy as np
from scipy.sparse import coo_matrix, eye
from sp_normalize import *
from pagerank import *
from QGC import *
import logging
import logging.config
import time
""" ---------------- init log config -------------------"""
log = logging.getLogger('test')
log.setLevel(logging.DEBUG)

""" file_hdlr = logging.FileHandler('log/QGC_{0}.log'.format(time.time()))"""
file_hdlr = logging.FileHandler('log/QGC_0.log')
file_hdlr.setLevel(logging.DEBUG)

console_hdlr = logging.StreamHandler()
console_hdlr.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)-8s - %(asctime)s - %(name)-12s - %(message)s')
file_hdlr.setFormatter(formatter)
console_hdlr.setFormatter(formatter)

log.addHandler(file_hdlr)
log.addHandler(console_hdlr)

log.info('QGC start')
# ----------------------------------------------------


alpha = 0.1
t = 10
maxitr = 200

""" Test Type => 1: toy graph; 2: UIT graph; 3: UIT graph; 4: DBLP graph """
test_type = 1

""" Re-read => 0: not re-read; 1: re-read """
# reread = 1

K = 3
topN = 0

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
list_u = list()
list_i = list()
list_r = list()
if test_type == 1:
    with open('TestData/simpleGraph_new.txt') as f:
        for line in f:
            ls = line.split()
            list_u.append(float(ls[0])-1)
            list_i.append(float(ls[1])-1)
            list_r.append(1.)
log.info('list_u : {0}'.format(list_u))
log.info('list_i : {0}'.format(list_i))

n = int(max(max(list_u), max(list_i))+1)
e = len(list_u)

matG = coo_matrix((list_r, (list_u, list_i)), shape=(n, n)).tocsr()
matG = matG + matG.transpose()

""" Create symmetric normalized matrix A of G """
matA, matB = SCNomalization(matG)


matI = eye(n, n)
xi = 0.2
matA = (1-xi) * matA + xi * matI
# print(matA) # 看起來應該是跟matlab 結果一樣的～


""" Test """
if test_type == 1:
    queries = [0]

""" query | Tree_NCut | L2Norm_NCut | KLD_NCut | Tree_topN | L2Norm_topN | KLD_topN """
# cellResult = cell(length(queries), 19);


""" Set the queries and their relevance """
for query in queries:

    if query < 0:
        continue

    """ if the query is isolated, ignore it. """
    # if q >= 0 and sum(matA[q, :]) == 0.2:
    #     continue

    """ Get the relevance vector """
    if query >= 0:
        vecRel = oneQ_pagerank(matB, query, 0.5)
        vecRel[query, 0] = 0
    else:
        vecRel = np.ones((n, 0)) / n

    if sum(vecRel) == 0:
        continue
    log.info('shape of vecRel: {0}'.format(vecRel.shape))
    """ 確定與 matlab 結果一樣～ """
    # for i in vecRel:
    #     print(np.round(float(i), 8))

    """ Algorithm: QGC5_QGC(Query-oriented spectral clustering)"""
    if act_QOGC_QGC == 1:
        log.info('Run Query-oriented spectral clustering algorithm...')
        # QGC_batch(matG, maxitr, q, K, 3, vecRel, 0, -0.02, 100)
        QGC_batch(matG, maxitr, vecRel, query, K, 3, topN)
    #     [iPhi, vecQ, vecPerform1, QGC_vecBestPerform, QGC_NCut_BestRel, vecPerform2, QGC_vecBestPerform2,
    #      QGC_NCut_BestRel2, vecPerform3, QGC_vecBestPerform3, QGC_NCut_BestRel3] = QOGC_QGC_batch(query, K, 3, topN)
    #     c_size1 = sum(full(vecPerform1))
    #     c_size2 = sum(full(vecPerform2))
    #     c_size3 = sum(full(vecPerform3))
    #     entropy_QGC1 = funcEntropy(c_size1
    #     ');
    #     entropy_QGC2 = funcEntropy(c_size2
    #     ');
    #     entropy_QGC3 = funcEntropy(c_size3
    #     ');
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