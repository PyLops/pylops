from itertools import chain

import numpy

from pylops.utils.backend import get_module


def _sampler_gaussian(m, batch_size, backend_module=numpy):
    return backend_module.random.randn(m, batch_size)


def _sampler_rayleigh(m, batch_size, backend_module=numpy):
    z = backend_module.random.randn(m, batch_size)
    for i in range(batch_size):
        z[:, i] *= m / backend_module.dot(z[:, i].T, z[:, i])
    return z


def _sampler_rademacher(m, batch_size, backend_module=numpy):
    return 2 * backend_module.random.binomial(1, 0.5, size=(m, batch_size)) - 1


_SAMPLERS = {
    "gaussian": _sampler_gaussian,
    "rayleigh": _sampler_rayleigh,
    "rademacher": _sampler_rademacher,
}


def trace_hutchinson(
    Op, neval=None, batch_size=None, sampler="rademacher", backend="numpy"
):
    r"""Trace of linear operator using the Hutchinson method.

    Returns an estimate of the trace of a linear operator using the Hutchinson
    method [1]_.

    Parameters
    ----------
    neval : :obj:`int`, optional
        Maximum number of matrix-vector products compute. Defaults to 10%
        of ``shape[1]``.
    batch_size : :obj:`int`, optional
        Vectorize computations by sampling sketching matrices instead of
        vectors. Set this value to as high as permitted by memory, but there is
        no guarantee of speedup. Coerced to never exceed ``neval``. When using
        "unitvector" as sampler, is coerced to not exceed ``shape[1]``.
        Defaults to 100 or ``neval``.
    sampler : :obj:`str`, optional
        Sample sketching matrices from the following distributions:

            - "gaussian": Mean zero, unit variance Gaussian.
            - "rayleigh": Sample from mean zero, unit variance Gaussian and
              normalize the columns.
            - "rademacher": Random sign.
            - "unitvector": Samples from the unit vectors :math:`\mathrm{e}_i`
              without replacement.

    backend : :obj:`str`, optional
        Backend used to densify matrix (``numpy`` or ``cupy``). Note that
        this must be consistent with how the operator has been created.

    Returns
    -------
    trace : :obj:`self.dtype`
        Operator trace.

    Raises
    -------
    ValueError
        If ``neval`` is smaller than 3.

    NotImplementedError
        If the ``sampler`` is not one of the available samplers.

    Notes
    -----
    Let :math:`m` = ``shape[1]`` and :math:`k` = ``neval``. This algorithm
    estimates the trace via

    .. math::
        \frac{1}{k}\sum\limits_{i=1}^k \mathbf{z}_i^T\,\mathbf{Op}\,\mathbf{z}_i

    where vectors :math:`\mathbf{z}_i` are sampled according to the sampling
    function. See [2]_ for a description of the variance and
    :math:`\epsilon`-approximation of different samplers.

    Prefer the Rademacher sampler if the goal is to minimize variance, but the
    Gaussian for a better probability of approximating the correct value. Use
    the Unit Vector approach if you are sampling a large number of ``neval``
    (compared to ``shape[1]``), especially if the operator is highly-structured.

    .. [1] Hutchinson, M. F. (1990). *A stochastic estimator of the trace of
           the influence matrix for laplacian smoothing splines*.
           Communications in Statistics - Simulation and Computation, 19(2),
           433–450.
    .. [2] Avron, H., and Toledo, S. (2011). *Randomized algorithms for
           estimating the trace of an implicit symmetric positive semi-definite
           matrix*. Journal of the ACM, 58(2), 1–34.
    """
    ncp = get_module(backend)
    m = Op.shape[1]
    neval = int(numpy.round(m * 0.1)) if neval is None else neval
    batch_size = min(neval, 100 if batch_size is None else batch_size)

    n_missing = neval - batch_size * (neval // batch_size)
    batch_range = chain(
        (batch_size for _ in range(0, neval - n_missing, batch_size)),
        (n_missing for _ in range(int(n_missing != 0))),
    )

    trace = ncp.zeros(1, dtype=Op.dtype)

    if sampler == "unitvector":
        remaining_vectors = list(range(m))
        n_total = 0
        while remaining_vectors:
            batch = min(batch_size, len(remaining_vectors))
            z = ncp.zeros((m, batch), dtype=Op.dtype)
            z_idx = ncp.random.choice(remaining_vectors, batch, replace=False)
            for i, idx in enumerate(z_idx):
                z[idx, i] = 1.0
                remaining_vectors.remove(idx)
            trace += ncp.trace((z.T @ (Op @ z)))
            n_total += batch
        trace *= m / n_total
        return trace[0]

    if sampler not in _SAMPLERS:
        raise NotImplementedError(f"sampler {sampler} not available.")

    sampler_fun = _SAMPLERS[sampler]
    for batch in batch_range:
        z = sampler_fun(m, batch, backend_module=ncp).astype(Op.dtype)
        trace += ncp.trace((z.T @ (Op @ z)))
    trace /= neval
    return trace[0]


def trace_hutchpp(Op, neval=None, sampler="rademacher", backend="numpy"):
    r"""Trace of linear operator using the Hutch++ method.

    Returns an estimate of the trace of a linear operator using the Hutch++
    method [1]_.

    Parameters
    ----------
    neval : :obj:`int`, optional
        Maximum number of matrix-vector products compute. Defaults to 10%
        of ``shape[1]``.
    sampler : :obj:`str`, optional
        Sample sketching matrices from the following distributions:

            - "gaussian": Mean zero, unit variance Gaussian.
            - "rayleigh": Sample from mean zero, unit variance Gaussian and
              normalize the columns.
            - "rademacher": Random sign.

    backend : :obj:`str`, optional
        Backend used to densify matrix (``numpy`` or ``cupy``). Note that
        this must be consistent with how the operator has been created.

    Returns
    -------
    trace : :obj:`self.dtype`
        Operator trace.

    Raises
    -------
    ValueError
        If ``neval`` is smaller than 3.

    NotImplementedError
        If the ``sampler`` is not one of the available samplers.

    Notes
    -----
    This function follows Algorithm 1 of [1]_. Let :math:`m` = ``shape[1]``
    and :math:`k` = ``neval``.

        1. Sample sketching matrices
           :math:`\mathbf{S} \in \mathbb{R}^{m \times \lfloor k/3\rfloor}`,
           and
           :math:`\mathbf{G} \in \mathbb{R}^{m \times \lfloor k/3\rfloor}`,
           from sub-Gaussian distributions.
        2. Compute reduced QR decomposition of :math:`\mathbf{Op}\,\mathbf{S}`,
           retaining only :math:`\mathbf{Q}`.
        3. Return :math:`\operatorname{tr}(\mathbf{Q}^T\,\mathbf{Op}\,\mathbf{Q}) + \frac{1}{\lfloor k/3\rfloor}\operatorname{tr}\left(\mathbf{G}^T(\mathbf{I} - \mathbf{Q}\mathbf{Q}^T)\,\mathbf{Op}\,(\mathbf{I} - \mathbf{Q}\mathbf{Q}^T)\mathbf{G}\right)`

    Use the Rademacher sampler unless you know what you are doing.

    .. [1] Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2021).
        *Hutch++: Optimal Stochastic Trace Estimation*. In Symposium on Simplicity
        in Algorithms (SOSA) (pp. 142–155). Philadelphia, PA: Society for
        Industrial and Applied Mathematics. `link <https://arxiv.org/abs/2010.09649>`_
    """

    ncp = get_module(backend)
    m = Op.shape[1]

    neval = int(numpy.round(m * 0.1)) if neval is None else neval

    if sampler not in _SAMPLERS:
        raise NotImplementedError(f"sampler {sampler} not available.")

    sampler_fun = _SAMPLERS[sampler]

    batch = neval // 3
    if batch <= 0:
        msg = f"Sampler '{sampler}' not supported with {neval} samples."
        msg += " Try increasing it."
        raise ValueError(msg)

    S = sampler_fun(m, batch, backend_module=ncp).astype(Op.dtype)
    G = sampler_fun(m, batch, backend_module=ncp).astype(Op.dtype)

    Q, _ = ncp.linalg.qr(Op @ S)
    del S
    G = G - Q @ (Q.T @ G)

    trace = ncp.zeros(1, dtype=Op.dtype)
    trace += ncp.trace(Q.T @ (Op @ Q)) + ncp.trace(G.T @ (Op @ G)) / batch
    return trace[0]


def trace_nahutchpp(
    Op,
    neval=None,
    sampler="rademacher",
    c1=1.0 / 6.0,
    c2=1.0 / 3.0,
    backend="numpy",
):
    r"""Trace of linear operator using the NA-Hutch++ method.

    Returns an estimate of the trace of a linear operator using the
    Non-Adaptive variant of Hutch++ method [1]_.

    Parameters
    ----------
    neval : :obj:`int`, optional
        Maximum number of matrix-vector products compute. Defaults to 10%
        of ``shape[1]``.
    sampler : :obj:`str`, optional
        Sample sketching matrices from the following distributions:

            - "gaussian": Mean zero, unit variance Gaussian.
            - "rayleigh": Sample from mean zero, unit variance Gaussian and
              normalize the columns.
            - "rademacher": Random sign.

    c1 : :obj:`float`, optional
        Fraction of ``neval`` for sketching matrix :math:`\mathbf{S}`.
    c2 : :obj:`float`, optional
        Fraction of ``neval`` for sketching matrix :math:`\mathbf{R}`. Must be
        larger than ``c2``, ideally by a factor of at least 2.
    backend : :obj:`str`, optional
        Backend used to densify matrix (``numpy`` or ``cupy``). Note that
        this must be consistent with how the operator has been created.

    Returns
    -------
    trace : :obj:`self.dtype`
        Operator trace.

    Raises
    -------
    ValueError
        If ``neval`` not large enough to accomodate ``c1`` and ``c2``.

    NotImplementedError
        If the ``sampler`` is not one of the available samplers.

    Notes
    -----
    This function follows Algorithm 2 of [1]_. Let :math:`m` = ``shape[1]``
    and :math:`k` = ``neval``.

        1. Fix constants :math:`c_1`, :math:`c_2`, :math:`c_3` such that
           :math:`c_1 < c_2` and :math:`c_1 + c_2 + c_3 = 1`.
        2. Sample sketching matrices
           :math:`\mathbf{S} \in \mathbb{R}^{m \times c_1 k}`,
           :math:`\mathbf{R} \in \mathbb{R}^{m \times c_2 k}`,
           and
           :math:`\mathbf{G} \in \mathbb{R}^{m \times c_3 k}`
           from sub-Gaussian distributions.
        3. Compute :math:`\mathbf{Z} = \mathbf{Op}\,\mathbf{R}`,
           :math:`\mathbf{W} = \mathbf{Op}\,\mathbf{S}`, and
           :math:`\mathbf{Y} = (\mathbf{S}^T \mathbf{Z})^+`, where :math:`+`
           denotes the Moore–Penrose inverse.
        4. Return :math:`\operatorname{tr}(\mathbf{Y} \mathbf{W}^T \mathbf{Z}) + \frac{1}{c_3 k} \left[ \operatorname{tr}(\mathbf{G}^T\,\mathbf{Op}\,\mathbf{G}) - \operatorname{tr}(\mathbf{G}^T\mathbf{Z}\mathbf{Y}\mathbf{W}^T\mathbf{G})\right]`

    The default values for :math:`c_1` and :math:`c_2` are set to :math:`1/6`
    and :math:`1/3`, respectively, but [1]_ suggests :math:`1/4` and :math:`1/2`.

    Use the Rademacher sampler unless you know what you are doing.

    .. [1] Meyer, R. A., Musco, C., Musco, C., & Woodruff, D. P. (2021).
        *Hutch++: Optimal Stochastic Trace Estimation*. In Symposium on Simplicity
        in Algorithms (SOSA) (pp. 142–155). Philadelphia, PA: Society for
        Industrial and Applied Mathematics. `link <https://arxiv.org/abs/2010.09649>`_
    """

    ncp = get_module(backend)
    m = Op.shape[1]
    neval = int(numpy.round(m * 0.1)) if neval is None else neval

    if sampler not in _SAMPLERS:
        raise NotImplementedError(f"sampler {sampler} not available.")

    sampler_fun = _SAMPLERS[sampler]

    batch1 = int(numpy.round(neval * c1))
    batch2 = int(numpy.round(neval * c2))
    batch3 = neval - batch1 - batch2
    if batch1 <= 0 or batch2 <= 0 or batch3 <= 0:
        msg = f"Sampler '{sampler}' not supported with {neval} samples."
        msg += " Try increasing it."
        raise ValueError(msg)

    S = sampler_fun(m, batch1, backend_module=ncp).astype(Op.dtype)
    R = sampler_fun(m, batch2, backend_module=ncp).astype(Op.dtype)
    G = sampler_fun(m, batch3, backend_module=ncp).astype(Op.dtype)

    Z = Op @ R
    Wt = (Op @ S).T
    Y = ncp.linalg.pinv(S.T @ Z)
    trace = ncp.zeros(1, dtype=Op.dtype)
    trace += (
        ncp.trace(Y @ Wt @ Z)
        + (ncp.trace(G.T @ (Op @ G)) - ncp.trace(G.T @ Z @ Y @ Wt @ G)) / batch3
    )
    return trace[0]
