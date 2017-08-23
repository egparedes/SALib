#################################################################################
  # Authors: 
  # First version by: Eunhye Song, Barry L. Nelson, Jeremy Staum (Northwestern University), 2015
  # Modified for 'sensitivity' package by: Bertrand Iooss (EDF R&D), 2017
#################################################################################

from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from scipy.stats import norm, gaussian_kde, rankdata

from . import common_args
from ..util import read_param_file


# https://github.com/cran/sensitivity/blob/master/R/shapleyPermEx.R
# https://rdrr.io/cran/sensitivity/man/shapleyPermEx.html
# http://mathesaurus.sourceforge.net/r-numpy.html


# Xall <- function(n) matrix(runif(d*n,-pi,pi),nc=d)
def x_all_fn(n):
    return np.random.rand([n, d]) * 2 * np.pi - np.pi


def x_set_fn(n, Sj, Sjc, xjc):
    return np.random.rand([n, Sj]) * 2 * np.pi - np.pi


def sample(problem, x_all_fn, x_set_fn, Nv, No, Ni):
    """Generates model inputs for computation of Shapley effects.

    Returns a NumPy matrix containing the model inputs. These model inputs are
    intended to be used with :func:`SALib.analyze.shapley.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    x_all_fn : (n) -> NumPy array
        A function to generate a n-sample of a d-dimensional input vector
    x_set_fn : (n, Sj, Sjc, xjc) -> NumPy array
        A function to generate a n- sample an input vector corresponding to the
        indices in Sj conditional on the input values xjc with the index set Sjc
    Nv : int
        Monte Carlo (MC) sample size to estimate the output variance
    No : int
        Output MC sample size to estimate the cost function
    Ni : int
        Inner MC sample size to estimate the cost function
    """

    D = problem['num_vars']

    perms = np.asarray(list(itertools.permutations(range(0, D))))
    m = perms.shape[0]
    X = np.empty([Nv + m * (D - 1) * No * Ni, D])
    X[:Nv, :] = x_all_fn(Nv, D)

    for p in range(0, m):
        pi = perms[p]
        pi_s = np.argsort(pi)

        for j in range(1, D - 1):
            Sj = pi[0:j]  # set of the 1st-jth elements in pi
            Sjc = pi[j:]  # set of the (j+1)th-dth elements in pi

            xjcM = np.reshape(x_set_fn(No, Sjc, None, None), [No, -1])  # sampled values of the inputs in Sjc
            for l in range(0, No):
                xjc = xjcM[l]

                # sample values of inputs in Sj conditional on xjc
                xj = x_set_fn(Ni, Sj, Sjc, xjc)
                xx = np.concatenate((xj, np.ones([Ni, len(xjc)]) * xjc), axis=1)
                start = Nv + (p)*(D-1)*No*Ni + (j-1)*No*Ni+(l)*Ni
                end = Nv + (p)*(D-1)*No*Ni + (j-1)*No*Ni+(l+1)*Ni
                X[start:end, :] = xx[:, pi_s]

    return X


def analyze(problem, X, Y, perms, Nv, No, Ni,
            print_to_console=False):
    """Perform Shapley effects analysis on model outputs.

    Returns a dictionary with keys 'mu', 'sigma' and 'Sh', 'S1', 'ST' where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.

    Parameters
    ----------
    problem : dict
        The problem definition
    X: numpy.matrix
        A NumPy matrix containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    Nv : int
        Monte Carlo (MC) sample size to estimate the output variance
    print_to_console : bool
        Print results directly to console (default False)

    References
    ----------
    .. [1] B. Iooss and C. Prieur, 2017, Analyse de sensibilite avec entrees
           dependantes : estimation par echantillonnage et par metamodeles des
           indices de Shapley,
           Submitted to 49emes Journees de la SFdS, Avignon, France, May 2017.

    .. [2] S. Kucherenko, S. Tarantola, and P. Annoni, 2012, Estimation of global
           sensitivity indices for models with dependent variables,
           Computer Physics Communications, 183, 937–946.

    .. [3] A.B. Owen, 2014, Sobol' indices and Shapley value,
           SIAM/ASA Journal of Uncertainty Quantification, 2, 245–251.

    .. [4] A.B. Owen and C. Prieur, 2016, On Shapley value for measuring
           importance of dependent inputs, submitted.

    .. [5] E. Song, B.L. Nelson, and J. Staum, 2016, Shapley effects for global
           sensitivity analysis: Theory and computation, SIAM/ASA Journal of
           Uncertainty Quantification, 4, 1060–1083.

    Examples
    --------
    >>> X = shapley.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = shapley.analyze(problem, X, Y, print_to_console=True)
    """

    D = problem['num_vars']

    # Initialize Shapley value for all players
    Sh = np.zeros(D)
    Sh2 = np.zeros(D)

    # Initialize main and total (Sobol) effects for all players
    Vsob = np.zeros(D)
    # Vsob2 = np.zeros(D)
    Tsob = np.zeros(D)
    # Tsob2 = np.zeros(D)

    # Estimate Var[Y]
    EY = np.mean(Y[:Nv])
    VarY = np.var(Y[:Nv])
    current_Y_idx = Nv

    # Estimate Shapley effects
    m = perms.shape[0]
    for p in range(0, m):
        pi = perms[p]
        prevC = 0
        for j in range(0, D):
            if j == D - 1:
                Chat = VarY
                Vsob[pi[j]] = Vsob[pi[j]] + prevC  # first order effect
                # Vsob2[pi[j]] = Vsob2[pi[j]] + prevC**2
            else:
                cVar = np.zeros(No)
                for l in range(0, No):
                    cVar[l] = np.var(Y[current_Y_idx:current_Y_idx + Ni], ddof=1)
                    current_Y_idx += Ni

                Chat = np.mean(cVar)

            dele = Chat - prevC

            Sh[pi[j]] = Sh[pi[j]] + dele
            Sh2[pi[j]] = Sh2[pi[j]] + dele**2

            prevC = Chat

            if j == 0:
                Tsob[pi[j]] = Tsob[pi[j]] + Chat  # Total effect
                # Tsob2[pi[j]] = Tsob2[pi[j]] + Chat**2

    Sh = Sh / m / VarY
    # Sh2 = Sh2 / m / VarY**2
    # ShSE = np.sqrt((Sh2 - Sh**2) / m)

    Vsob = Vsob / (m/D) / VarY  # averaging by number of permutations with j=d-1
    # Vsob2 = Vsob2 / (m/D) / VarY**2
    # VsobSE = np.sqrt((Vsob2 - Vsob**2) / (m/D))
    Vsob = 1 - Vsob
    # Vsob2 = 1 - Vsob2

    Tsob = Tsob / (m/D) / VarY  # averaging by number of permutations with j=1
    # Tsob2 = Tsob2 / (m/D) / VarY**2
    # TsobSE = np.sqrt((Tsob2 - Tsob**2) / (m/D))

    results = dict(mu=EY, sigma=VarY, Sh=Sh, S1=Vsob, ST=Tsob)

    if print_to_console:
        for i in range(D):
            print("[%s] Shapley: %f | Sobol: %f | Total Sobol: %f" % (problem['names'][i], 
                                                                      results['Sh'][i],
                                                                      results['S1'][i],
                                                                      results['ST'][i]))

    return results
