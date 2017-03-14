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


def sigmoid(s):
    """ sigmoid function """
    theta = 1.0 / (1.0 + np.exp(-s))
    return theta


def matlab_sigmoid(s, a, c):
    """ sigmoid function """
    theta = 1.0 / (1.0 + np.exp(-a*(s-c)))
    return theta


def sigmoid_mat(mat):
    """ sigmf Matrix將矩陣數值轉為0.5~1之間的值，若無相關則為0 """
    vec_val = mat[mat.nonzero()]
    """ 非零的值超過1個才做，否則就使用原矩陣即可，使用.shape[1]避免矩陣格式不同而讀不到nnz的問題 """
    if vec_val.shape[1] > 0:
        vec_val = vec_val.tolist()[0]    # 轉為list後面才可計算sigmoid
        (vecpos_x, vecpos_y) = mat.nonzero()
        (m, n) = mat.shape
        vec_average = np.average(vec_val)
        val_sigmoid_scale = np.std(np.array(vec_val))
        """ avoid x/0 """
        if val_sigmoid_scale == 0:
            val_sigmoid_scale = 1
        else:
            """ remove outlier and calculate new std """
            std_vec_val = [x for x in vec_val if x < (val_sigmoid_scale + vec_average)]
            val_sigmoid_scale = np.std(np.array(std_vec_val))
            if val_sigmoid_scale == 0:
                val_sigmoid_scale = 1
        """ sigmoid """
        vec_val = sigmoid(np.array(vec_val) / val_sigmoid_scale)
        mat = coo_matrix((vec_val, (vecpos_x, vecpos_y)), shape=(m, n))
    return mat

if __name__ == '__main__':
    a = np.matrix([[0, 5, 3, 1],
                   [5, 0, 1, 0],
                   [3, 1, 0, 2],
                   [1, 0, 2, 0]])
    spa = lil_matrix(a)
    print(SymmetricNomalization(spa).todense())