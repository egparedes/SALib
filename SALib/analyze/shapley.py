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



def sample()

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
