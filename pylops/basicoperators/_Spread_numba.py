import os
from numba import jit, prange

# detect whether to use parallel or not
numba_threads = int(os.getenv('NUMBA_NUM_THREADS', '1'))
parallel = True if numba_threads != 1 else False


@jit(nopython=True, parallel=parallel, nogil=True)
def _matvec_numba_table(x, y, dims, interp, table, dtable):
    """numba implementation of forward mode with table.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dims)
    for ix0 in range(dim0):
        for it in range(dim1):
            indices = table[ix0, it]
            if interp:
                dindices = dtable[ix0, it]

            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[i, index] += x[ix0, it]
                    else:
                        y[i, index] += (1 -dindices[i])*x[ix0, it]
                        y[i, index + 1] += dindices[i] * x[ix0, it]
    return y.ravel()

@jit(nopython=True, parallel=parallel, nogil=True)
def _rmatvec_numba_table(x, y, dims, dimsd, interp, table, dtable):
    """numba implementation of adjoint mode with table.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dimsd)
    for ix0 in prange(dim0):
        for it in range(dim1):
            indices = table[ix0, it]
            if interp:
                dindices = dtable[ix0, it]

            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[ix0, it] += x[i, index]
                    else:
                        y[ix0, it] += x[i, index]*(1 - dindices[i]) + \
                                      x[i, index + 1]*dindices[i]
    return y.ravel()

@jit(nopython=True, parallel=parallel, nogil=True)
def _matvec_numba_onthefly(x, y, dims, interp, fh):
    """numba implementation of forward mode with on-the-fly computations.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dims)
    for ix0 in range(dim0):
        for it in range(dim1):
            if interp:
                indices, dindices = fh(ix0, it)
            else:
                indices, dindices = fh(ix0, it)
            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[i, index] += x[ix0, it]
                    else:
                        y[i, index] += (1 -dindices[i])*x[ix0, it]
                        y[i, index + 1] += dindices[i] * x[ix0, it]
    return y.ravel()

@jit(nopython=True, parallel=parallel, nogil=True)
def _rmatvec_numba_onthefly(x, y, dims, dimsd, interp, fh):
    """numba implementation of adjoint mode with on-the-fly computations.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dimsd)
    for ix0 in prange(dim0):
        for it in range(dim1):
            if interp:
                indices, dindices = fh(ix0, it)
            else:
                indices, dindices = fh(ix0, it)
            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[ix0, it] += x[i, index]
                    else:
                        y[ix0, it] += x[i, index]*(1 - dindices[i]) + \
                                      x[i, index + 1]*dindices[i]
    return y.ravel()