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

from io import BytesIO

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from PIL import Image

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
xnest = LRop / yn
noise_est = np.sqrt(np.sum((yn - LRop @ xnest) ** 2) / (N - 1))

###############################################################################
# Let's plot the best fitting line for the case of noise free and noisy data
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    np.array([t.min(), t.max()]),
    np.array([t.min(), t.max()]) * x[1] + x[0],
    "k",
    lw=4,
    label=rf"True: $x_0$ = {x[0]:.2f}, $x_1$ = {x[1]:.2f}",
)
ax.plot(
    np.array([t.min(), t.max()]),
    np.array([t.min(), t.max()]) * xnest[1] + xnest[0],
    "--g",
    lw=4,
    label=rf"MAP Estimated: $x_0$ = {xnest[0]:.2f}, $x_1$ = {xnest[1]:.2f}",
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
# Priors in Bayesian analysis can be interpreted as the probabilistic
# equivalent to regularization. Finding the maximum a posteriori (MAP) estimate
# to a least-squares problem with a Gaussian prior on the parameters is
# is equivalent to applying a Tikhonov (L2) regularization to these parameters.
# A Laplace prior is equivalent to a sparse (L1) regularization. In addition,
# the weight of the regularization is controlled by the "scale" of the
# distribution of the prior; the standard deviation (in the case of a Gaussian)
# is inversely proportional strength of the regularization.
#
# In this problem we will use weak, not very informative priors, by settings
# their "weights" to be small (high scale parameters):
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
    xp = pm.Normal("x", 0, sigma=20, shape=dims)
    mu = pm.Deterministic("mu", pytensor_lrop(xp))

    # Define likelihood
    likelihood = pm.Normal("y", mu=mu, sigma=sp, observed=y_data)

    # Inference!
    idata = pm.sample(2000, tune=1000, chains=2)

###############################################################################
# The PyMC library offers a way to visualize the Bayesian model which is
# helpful in mapping dependencies.
dot = pm.model_to_graphviz(model)

# Some magic to display in docs
plt.imshow(Image.open(BytesIO(dot.pipe("png"))))
plt.axis("off")

###############################################################################
axes = az.plot_trace(idata, figsize=(10, 7), var_names=["~mu"])
axes[0, 0].axvline(x[0], label="True Intercept", lw=2, color="k")
axes[0, 0].axvline(xnest[0], label="Intercept MAP", lw=2, color="C0", ls="--")
axes[0, 0].axvline(x[1], label="True Slope", lw=2, color="darkgray")
axes[0, 0].axvline(xnest[1], label="Slope MAP", lw=2, color="C1", ls="--")
axes[0, 1].axhline(x[0], label="True Intercept", lw=2, color="k")
axes[0, 1].axhline(xnest[0], label="Intercept MAP", lw=2, color="C0", ls="--")
axes[0, 1].axhline(x[1], label="True Slope", lw=2, color="darkgray")
axes[0, 1].axhline(xnest[1], label="Slope MAP", lw=2, color="C1", ls="--")
axes[1, 0].axvline(sigma, label="True Sigma", lw=2, color="k")
axes[1, 0].axvline(noise_est, label="Sigma MAP", lw=2, color="C0", ls="--")
axes[1, 1].axhline(sigma, label="True Sigma", lw=2, color="k")
axes[1, 1].axhline(noise_est, label="Sigma MAP", lw=2, color="C0", ls="--")
for ax in axes.ravel():
    ax.legend()
ax.get_figure().tight_layout()

################################################################################
# The plot above is known as the "trace" plot. The plots on the left display the
# posterior distribution of all latent variables in the model. The top-left
# has multiple colored, one for each parameter in the latent vector
# :math:`\mathbf{x}`. The bottom left plot displays the posterior of the
# estimated noise.
#
# In these plots there are multiple distributions of the same color. Each
# represents a "chain". A chain is a single run of a Monte Carlo algorithm.
# Generally, Monte Carlo methods run various chains to ensure that all regions
# of the posterior distribution are sampled. These chains are shown on the right
# hand plots.
#
# With this model, we can obtain an uncertainty measurement via the High Density
# Interval. To do that, we need to sample the "preditive posterior", that is,
# the posterior distribution of the data, given the model.

with model:
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

###############################################################################
fig, ax = plt.subplots(figsize=(8, 4))
az.plot_hdi(
    t,
    idata.posterior_predictive["y"],
    fill_kwargs={"label": "95% HDI"},
    hdi_prob=0.95,
    ax=ax,
)
ax.plot(
    np.array([t.min(), t.max()]),
    np.array([t.min(), t.max()]) * x[1] + x[0],
    "k",
    lw=4,
    label=rf"True: $x_0$ = {x[0]:.2f}, $x_1$ = {x[1]:.2f}",
)
ax.plot(
    np.array([t.min(), t.max()]),
    np.array([t.min(), t.max()]) * xnest[1] + xnest[0],
    "--g",
    lw=4,
    label=rf"MAP Estimated: $x_0$ = {xnest[0]:.2f}, $x_1$ = {xnest[1]:.2f}",
)
ax.scatter(t, y, c="r", s=70)
ax.scatter(t, yn, c="g", s=70)
ax.legend()
fig.tight_layout()
