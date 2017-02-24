import numpy as np
from scipy.sparse import lil_matrix, coo_matrix

""" --- Column-normalization --- """
def ColumnNomalization(matrix):
    valVeretxIDMax = matrix.shape[0]
    vecOne = np.ones((1, valVeretxIDMax))
    """ 取出每行的維度與index"""
    vecColDegree = vecOne * matrix
    vecVal = vecColDegree[vecColDegree.nonzero()].tolist()
    (vecPosX, vecPosY) = vecColDegree.nonzero()
    """ 將所有的值改為 1 / 維度  (PageRank )"""
    vecVal = [1. / float(x) for x in vecVal]
    # print('[ColumnNomalization]vecVal',vecVal)
    vecPosY = vecPosY.tolist()
    """ 做出對角矩陣 值為 1 / 每行的維度 """
    matrixD = coo_matrix((vecVal, (vecPosY, vecPosY)), shape=(valVeretxIDMax, valVeretxIDMax))
    """ 將原來矩陣所有的值做 normalization ( 值 *  1 / 每行的維度 = 對於行的比重?) """
    # print('[ColumnNomalization] matrixD',matrixD.todense())
    matrixRet = matrix * matrixD
    # print('[ColumnNomalization] matrixRet',matrixRet.todense())
    return matrixRet


""" --- Row-normalization --- """
def RowNomalization(matrix):
    valVeretxIDMax = matrix.shape[1]
    vecOne = np.ones((valVeretxIDMax, 1))
    vecRowDegree = matrix * vecOne
    vecVal = vecRowDegree[vecRowDegree.nonzero()].tolist()
    (vecPosX, vecPosY) = vecRowDegree.nonzero()
    vecVal = [1. / x for x in vecVal]
    vecPosX = vecPosX.tolist()

    matrixD = coo_matrix((vecVal, (vecPosX, vecPosX)), shape=(valVeretxIDMax, valVeretxIDMax))

    matrixRet = matrixD * matrix
    return matrixRet

""" --- Symmetric-normalization --- """
def SymmetricNomalization(matrix):
    valVeretxIDMax = matrix.shape[1]
    vecOne = np.ones((valVeretxIDMax, 1))
    vecRowDegree = matrix * vecOne
    vecVal = vecRowDegree[vecRowDegree.nonzero()].tolist()
    (vecPosX, vecPosY) = vecRowDegree.nonzero()
    vecVal = [1. / (x ** 0.5) for x in vecVal]
    vecPosX = vecPosX.tolist()

    matrixD = coo_matrix((vecVal, (vecPosX, vecPosX)), shape=(valVeretxIDMax, valVeretxIDMax))

    matrixRet = matrixD * matrix * matrixD
    return matrixRet

def SCNomalization(matrix):
    valVeretxIDMax = matrix.shape[1]
    vecOne = np.ones((valVeretxIDMax, 1))
    vecRowDegree = matrix * vecOne
    vecVal = vecRowDegree[vecRowDegree.nonzero()].tolist()
    (vecPosX, vecPosY) = vecRowDegree.nonzero()
    vecVal = [1. / (x ** 0.5) for x in vecVal]
    vecPosX = vecPosX.tolist()

    matrixD = coo_matrix((vecVal, (vecPosX, vecPosX)), shape=(valVeretxIDMax, valVeretxIDMax)).tocsr()

    matrixRet = matrixD * matrix * matrixD
    matrixC = matrix * matrixD * matrixD
    return matrixRet, matrixC


if __name__ == '__main__':
    a = np.matrix([[0, 5, 3, 1],
                   [5, 0, 1, 0],
                   [3, 1, 0, 2],
                   [1, 0, 2, 0]])
    spa = lil_matrix(a)
    print(SymmetricNomalization(spa).todense())