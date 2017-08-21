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
    X[:Nv, :] = x_all_fn(Nv)

    for p in range(0, m):
        pi = perms[p]
        pi_s = np.argsort(pi)

        for j in range(0, d - 1):
            Sj = pi[:j]  # set of the 1st-jth elements in pi
            Sjc = pi[j:]  # set of the (j+1)th-dth elements in pi

            xjcM = np.reshape(x_set_fn(No, Sjc, None, None), [No, -1])  # sampled values of the inputs in Sjc
            for l in range(0, No):
                xjc = xjcM[l]

                # sample values of inputs in Sj conditional on xjc
                xj = x_set_fn(Ni, Sj, Sjc, xjc)
                # xx <- cbind(xj, matrix(xjc,nrow=Ni,ncol=length(xjc),byrow=T))
                xx = np.concatenate((xj, np.reshape(xjc, [Ni, -1])), axis=1)
                # X[(Nv+(p-1)*(d-1)*No*Ni+(j-1)*No*Ni+(l-1)*Ni+1):(Nv+(p-1)*(d-1)*No*Ni+(j-1)*No*Ni+l*Ni),] <- xx[,pi_s]
                X[ Nv + (p)*(d-1)*No*Ni + (j)*No*Ni+(l)*Ni + 1: Nv + (p)*(d)*No*Ni + (j)*No*Ni+l*Ni, :] = xx[:, pi_s]

    return X


def analyze(problem,
            X, Y, x_all_fn, x_set_fn, Nv, No, Ni,
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
    x_all_fn: (n) -> NumPy array
        A function to generate a n-sample of a d-dimensional input vector
    x_set_fn: (n, Sj, Sjc, xjc) -> NumPy array
        A function to generate a n- sample an input vector corresponding to the
        indices in Sj conditional on the input values xjc with the index set Sjc
    Nv: int
        Monte Carlo (MC) sample size to estimate the output variance
    No: int
        Output MC sample size to estimate the cost function
    Ni: int
        Inner MC sample size to estimate the cost function
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
    >>> X = latin.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = delta.analyze(problem, X, Y, print_to_console=True)
    """

    D = problem['num_vars']
    N = Y.size

    perms = np.asarray(list(itertools.permutations(range(0, 4))))
    m = perms.shape[0]

    values = np.empty([])
    X <- matrix(NA, ncol=d, nrow=Nv+m*(d-1)*No*Ni)
    X[1:Nv,] <- Xall(Nv)




    keys = ('mu', 'sigma', 'Sh', 'S1', 'ST')
    S = dict((k, np.zeros(D)) for k in keys)
    # if print_to_console:
    #     print("Parameter %s %s %s %s" % keys)

    # for i in range(D):
    #     S['delta'][i], S['delta_conf'][i] = bias_reduced_delta(
    #         Y, Ygrid, X[:, i], m, num_resamples, conf_level)
    #     S['S1'][i] = sobol_first(Y, X[:, i], m)
    #     S['S1_conf'][i] = sobol_first_conf(
    #         Y, X[:, i], m, num_resamples, conf_level)
    #     if print_to_console:
    #         print("%s %f %f %f %f" % (problem['names'][i], S['delta'][
    #               i], S['delta_conf'][i], S['S1'][i], S['S1_conf'][i]))

    return S
