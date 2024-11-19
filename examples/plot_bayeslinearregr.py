r"""
Bayesian Linear Regression
==========================

In the :ref:`sphx_glr_gallery_plot_linearregr.py` example, we
performed linear regression by applying a variety of solvers to the
:py:class:`pylops.LinearRegression` operator.

In this example, we will apply linear regression the Bayesian way.
In Bayesian inference, we are not looking for a "best" estimate
of the linear regression parameters; rather, we are looking for
all possible parameters and their associated (posterior) probability,
that is, how likely that those are the parameters that generated our data.

To do this, we will leverage the probabilistic programming library
`PyMC <https://www.pm.io>`_.

In the Bayesian formulation, we write the problem in the following manner:

    .. math::
        y_i \sim N(x_0 + x_1 t_i, \sigma)  \qquad \forall i=0,1,\ldots,N-1

where :math:`x_0` is the intercept and :math:`x_1` is the gradient.
This notation means that the obtained measurements :math:`y_i` are normally distributed around
mean :math:`x_0 + x_1 t_i` with a standard deviation of :math:`\sigma`.
We can also express this problem in a matrix form, which makes it clear that we
can use a PyLops operator to describe this relationship.

    .. math::
        \mathbf{y} \sim N(\mathbf{A} \mathbf{x}, \sigma)

In this example, we will combine the Bayesian power of PyMC with the linear language of
PyLops.
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

import pylops

plt.close("all")
np.random.seed(10)

###############################################################################
# Define the input parameters: number of samples along the t-axis (``N``),
# linear regression coefficients (``x``), and standard deviation of noise
# to be added to data (``sigma``).
N = 30
x = np.array([1.0, 0.5])
sigma = 0.25

###############################################################################
# Let's create the time axis and initialize the
# :py:class:`pylops.LinearRegression` operator
t = np.linspace(0, 1, N)
LRop = pylops.LinearRegression(t, dtype=t.dtype)

###############################################################################
# We can then apply the operator in forward mode to compute our data points
# along the x-axis (``y``). We will also generate some random gaussian noise
# and create a noisy version of the data (``yn``).
y = LRop @ x
yn = y + np.random.normal(0, sigma, N)

###############################################################################
# The deterministic solution is to solve the
# :math:`\mathbf{y} =  \mathbf{A} \mathbf{x}` in a least-squares sense.
# Using PyLops, the ``/`` operator solves the iteratively (i.e.,
# :py:func:`scipy.sparse.linalg.lsqr`).
# In Bayesian terminology, this estimator is known as the maximulum likelihood
# estimation (MLE).
x_mle = LRop / yn
noise_mle = np.sqrt(np.sum((yn - LRop @ x_mle) ** 2) / (N - 1))

###############################################################################
# Alternatively, we may regularize the problem. In this case we will condition
# the solution towards smaller magnitude parameters, we can use a regularized
# least squares approach. Since the weight is pretty small, we expect the
# result to be very similar to the one above.
sigma_prior = 20
eps = 1 / np.sqrt(2) / sigma_prior
x_map, *_ = pylops.optimization.basic.lsqr(LRop, yn, damp=eps)
noise_map = np.sqrt(np.sum((yn - LRop @ x_map) ** 2) / (N - 1))

###############################################################################
# Let's plot the best fitting line for the case of noise free and noisy data
fig, ax = plt.subplots(figsize=(8, 4))
for est, est_label, c in zip(
    [x, x_mle, x_map], ["True", "MLE", "MAP"], ["k", "C0", "C1"]
):
    ax.plot(
        np.array([t.min(), t.max()]),
        np.array([t.min(), t.max()]) * est[1] + est[0],
        color=c,
        ls="--" if est_label == "MAP" else "-",
        lw=4,
        label=rf"{est_label}: $x_0$ = {est[0]:.2f}, $x_1$ = {est[1]:.2f}",
    )
ax.scatter(t, y, c="r", s=70)
ax.scatter(t, yn, c="g", s=70)
ax.legend()
fig.tight_layout()

###############################################################################
# Let's solve this problem the Bayesian way, which consists in obtaining the
# posterior probability :math:`p(\mathbf{x}\,|\,\mathbf{y})` via Bayes theorem:
#
# .. math::
#    \underbrace{p(\mathbf{x} \,|\, \mathbf{y})}_{\text{posterior}}
#    \propto \overbrace{p(\mathbf{y} \,|\, \mathbf{x})}^{\text{likelihood}}\;
#    \overbrace{p(\mathbf{x})}^{\text{prior}}
#
# To do so, we need to define the priors and the likelihood.
#
# As hinted above, priors in Bayesian analysis can be interpreted as the
# probabilistic equivalent to regularization. Finding the maximum a posteriori
# (MAP) estimate to a least-squares problem with a Gaussian prior on the
# parameters is equivalent to applying a Tikhonov (L2) regularization to these
# parameters. A Laplace prior is equivalent to a sparse (L1) regularization.
# In addition, the weight of the regularization is controlled by the "scale" of
# the distribution of the prior; the standard deviation (in the case of a Gaussian)
# is inversely proportional strength of the regularization. So if we use the same
# sigma_prior above as the standard deviation of our prior distribition, we
# should get the same MAP out of them. In practice, in Bayesian analysis we are
# not only interested in point estimates like MAP, but rather, the whole
# posterior distribution. If you want the MAP only, there are better,
# methods to obtain them, such as the one shown above.
#
# In this problem we will use weak, not very informative priors, by setting
# their prior to accept a wide range of probable values. This is equivalent to
# setting the "weights" to be small, as shown above:
#
# .. math::
#     x_0 \sim N(0, 20)
#
#     x_1 \sim N(0, 20)
#
#     \sigma \sim \text{HalfCauchy}(10)
#
# The (log) likelihood in Bayesian analysis is the equivalent of the cost
# function in deterministic inverse problems. In this case we have already
# seen this likelihood:
#
# .. math::
#       p(\mathbf{y}\,|\,\mathbf{x}) \sim N(\mathbf{A}\mathbf{x}, \sigma)
#

# Construct a PyTensor `Op` which can be used in a PyMC model.
pytensor_lrop = pylops.PyTensorOperator(LRop)
dims = pytensor_lrop.dims  # Inherits dims, dimsd and shape from LRop

# Construct the PyMC model
with pm.Model() as model:
    y_data = pm.Data("y_data", yn)

    # Define priors
    sp = pm.HalfCauchy("σ", beta=10)
    xp = pm.Normal("x", 0, sigma=sigma_prior, shape=dims)
    mu = pm.Deterministic("mu", pytensor_lrop(xp))

    # Define likelihood
    likelihood = pm.Normal("y", mu=mu, sigma=sp, observed=y_data)

    # Inference!
    idata = pm.sample(500, tune=200, chains=2)

###############################################################################
# The plot below is known as the "trace" plot. The left column displays the
# posterior distributions of all latent variables in the model. The top-left
# plot has multiple colored posteriors, one for each parameter of the latent
# vector :math:`\mathbf{x}`. The bottom left plot displays the posterior of the
# estimated noise :math:`\sigma`.
#
# In these plots there are multiple distributions of the same color and
# multiple line styles. Each of these represents a "chain". A chain is a single
# run of a Monte Carlo algorithm. Generally, Monte Carlo methods run various
# chains to ensure that all regions of the posterior distribution are sampled.
# These chains are shown on the right hand plots.

axes = az.plot_trace(idata, figsize=(10, 7), var_names=["~mu"])
axes[0, 0].axvline(x[0], label="True Intercept", lw=2, color="k")
axes[0, 0].axvline(x_map[0], label="Intercept MAP", lw=2, color="C0", ls="--")
axes[0, 0].axvline(x[1], label="True Slope", lw=2, color="darkgray")
axes[0, 0].axvline(x_map[1], label="Slope MAP", lw=2, color="C1", ls="--")
axes[0, 1].axhline(x[0], label="True Intercept", lw=2, color="k")
axes[0, 1].axhline(x_map[0], label="Intercept MAP", lw=2, color="C0", ls="--")
axes[0, 1].axhline(x[1], label="True Slope", lw=2, color="darkgray")
axes[0, 1].axhline(x_map[1], label="Slope MAP", lw=2, color="C1", ls="--")
axes[1, 0].axvline(sigma, label="True Sigma", lw=2, color="k")
axes[1, 0].axvline(noise_map, label="Sigma MAP", lw=2, color="C0", ls="--")
axes[1, 1].axhline(sigma, label="True Sigma", lw=2, color="k")
axes[1, 1].axhline(noise_map, label="Sigma MAP", lw=2, color="C0", ls="--")
for ax in axes.ravel():
    ax.legend()
ax.get_figure().tight_layout()

################################################################################
# With this model, we can obtain an uncertainty measurement via the High Density
# Interval. To do that, we need to sample the "preditive posterior", that is,
# the posterior distribution of the data, given the model. What this does is
# sample the latent vetors from their posteriors (above), and use the model
# to construct realizations of the data given these realizations. They represent
# what the model thinks the data should look like, given everything it has
# already seen.

with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

###############################################################################
# sphinx_gallery_thumbnail_number = 3
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_hdi(
    t,
    idata.posterior_predictive["y"],
    fill_kwargs={"label": "95% HDI"},
    hdi_prob=0.95,
    ax=ax,
)
for est, est_label, c in zip(
    [x, x_mle, x_map], ["True", "MLE", "MAP"], ["k", "C0", "C1"]
):
    ax.plot(
        np.array([t.min(), t.max()]),
        np.array([t.min(), t.max()]) * est[1] + est[0],
        color=c,
        ls="--" if est_label == "MAP" else "-",
        lw=4,
        label=rf"{est_label}: $x_0$ = {est[0]:.2f}, $x_1$ = {est[1]:.2f}",
    )
ax.scatter(t, y, c="r", s=70)
ax.scatter(t, yn, c="g", s=70)
ax.legend()
fig.tight_layout()
