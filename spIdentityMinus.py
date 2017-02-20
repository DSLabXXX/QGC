import numpy as np
from scipy import sparse


def del_sp_row(mat, row_idx):
    size = mat.shape
    d_row = mk_eye(row_idx, size[0], 1)
    return d_row * mat


def del_sp_col(mat, col_idx):
    size = mat.shape
    d_col = mk_eye(col_idx, size[0], 2)
    return mat * d_col


def del_sp_row_col(mat, idx):
    size = mat.shape
    d_row = mk_eye(idx, size[0], 1)
    d_col = mk_eye(idx, size[0], 2)
    return d_row * mat * d_col


def mk_eye(idx, N, axis_type):
    # idx is row or col index for delete
    # N is shape[0] of mat
    # axis_type 1: row, 2: col
    x = list()
    y = list()
    value = list()
    if axis_type == 1:
        shape = (N-1, N)
    else:
        shape = (N, N-1)
    for i in range(N):
        if i < idx:
            x.append(i)
            y.append(i)
            value.append(1)
        elif i == idx:
            pass
        elif i > idx:
            if axis_type == 1:
                x.append(i-1)
                y.append(i)
            else:
                x.append(i)
                y.append(i-1)
            value.append(1)
    s = sparse.coo_matrix((value, (x, y)), shape=shape).tocsr()
    # print(s.todense())
    return s


def sp_insert_rows(mtx_a, mtx_to_insert, idx):
    mtx_tmp_first = mtx_a[:idx, :]
    mtx_tmp_last = mtx_a[idx:, :]
    return sparse.vstack([mtx_tmp_first, mtx_to_insert, mtx_tmp_last])


if __name__ == '__main__':
    a = sparse.lil_matrix(np.array([[1,2,3],[4,5,6],[7,8,9]])).tocsr()

    print('original:\n', a.todense())
    print('delete col&row by index=1:\n', del_sp_row_col(a, 1).todense())

    """ -----------------好像太花時間了 需要找時間改----------------- """
    """
        use 0.0046498775482177734s to make (9999, 10000) eye mat
        use 0.02309584617614746s to make (49999, 50000) eye mat
        use 0.046596527099609375s to make (99999, 100000) eye mat
        use 0.06904172897338867s to make (149999, 150000) eye mat
        use 0.0919654369354248s to make (199999, 200000) eye mat
        use 0.4411921501159668s to make (999999, 1000000) eye mat
        use 0.6989400386810303s to make (1499999, 1500000) eye mat
        use 4.358846187591553s to make (9999999, 10000000) eye mat
    """
    import time
    n = 200000
    st = time.time()
    mk_eye(2, n, 1)
    tt = time.time() - st
    print('use {0}s to make a {1} eye mat'.format(tt, (n-1, n)))
    """ -----------------好像太花時間了 需要找時間改----------------- """
