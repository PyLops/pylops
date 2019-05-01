from numba import jit, prange

@jit(nopython=True, parallel=True, nogil=True)
def _matvec_numba_table(x, y, dims, interp, table, dtable):
    """numba implementation of forward mode with table.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dims)
    for isp in range(dim0):
        for it in range(dim1):
            indices = table[isp, it]
            if interp:
                dindices = dtable[isp, it]

            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[i, index] += x[isp, it]
                    else:
                        y[i, index] += (1 -dindices[i])*x[isp, it]
                        y[i, index + 1] += dindices[i] * x[isp, it]
    return y.ravel()

@jit(nopython=True, parallel=True, nogil=True)
def _rmatvec_numba_table(x, y, dims, dimsd, interp, table, dtable):
    """numba implementation of adjoint mode with table.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dimsd)
    for isp in prange(dim0):
        for it in range(dim1):
            indices = table[isp, it]
            if interp:
                dindices = dtable[isp, it]

            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[isp, it] += x[i, index]
                    else:
                        y[isp, it] += x[i, index]*(1 - dindices[i]) + \
                                      x[i, index + 1]*dindices[i]
    return y.ravel()

@jit(nopython=True, parallel=True, nogil=True)
def _matvec_numba_onthefly(x, y, dims, interp, fh):
    """numba implementation of forward mode with on-the-fly computations.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dims)
    for isp in range(dim0):
        for it in range(dim1):
            if interp:
                indices, dindices = fh(isp, it)
            else:
                indices, dindices = fh(isp, it)
            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[i, index] += x[isp, it]
                    else:
                        y[i, index] += (1 -dindices[i])*x[isp, it]
                        y[i, index + 1] += dindices[i] * x[isp, it]
    return y.ravel()

@jit(nopython=True, parallel=True, nogil=True)
def _rmatvec_numba_onthefly(x, y, dims, dimsd, interp, fh):
    """numba implementation of adjoint mode with on-the-fly computations.
    See official documentation for description of variables
    """
    dim0, dim1 = dims
    x = x.reshape(dimsd)
    for isp in prange(dim0):
        for it in range(dim1):
            if interp:
                indices, dindices = fh(isp, it)
            else:
                indices, dindices = fh(isp, it)
            for i, indexfloat in enumerate(indices):
                index = int(indexfloat)
                if index != -9223372036854775808: # =int(np.nan)
                    if not interp:
                        y[isp, it] += x[i, index]
                    else:
                        y[isp, it] += x[i, index]*(1 - dindices[i]) + \
                                      x[i, index + 1]*dindices[i]
    return y.ravel()