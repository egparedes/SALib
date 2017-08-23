from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

# from scipy.stats import norm, gaussian_kde, rankdata

# from . import common_args
# from ..util import read_param_file

#
#   Source code adapted from R 'sensitivity' package:
#       https://github.com/cran/sensitivity/blob/master/R/shapleyPermEx.R
#
#   Authors:
#
#     * First version by:
#           Eunhye Song, Barry L. Nelson, Jeremy Staum (Northwestern University), 2015
#     * Modified for R 'sensitivity' package by:
#           Bertrand Iooss (EDF R&D), 2017
#     * Adapted for Python 'SALib' package by:
#           Enrique G. Paredes <egparedes@ifi.uzh.ch> (VMMLab, University of Zurich), 2017
#


def analyze(problem, X, Y, N_v, N_o, N_i, print_to_console=False):
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
    N_v : int
        Used Monte Carlo (MC) sample size to estimate the output variance
    N_o : int
        Used output MC sample size to estimate the cost function
    N_i : int
        Used inner MC sample size to estimate the cost function
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
    >>> problem = dict(num_vars=3, names=['x1', 'x2', 'x3'])
    >>> N_v, N_o, N_i = 10000, 1000, 3
    >>> X = shapley.sample(problem, N_v=N_v, N_o=N_o, N_i=N_i)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = shapley.analyze(problem, X, Y, N_v=N_v, N_o=N_o, N_i=N_i,
                             print_to_console=True)
    """

    D = problem['num_vars']
    perms = np.asarray(list(itertools.permutations(range(0, D))))
    m = perms.shape[0]

    # Initialize Shapley value for all players
    shapley_Y = np.zeros(D)
    # shapley_Y2 = np.zeros(D)

    # Initialize main and total (Sobol) effects for all players
    sobol_Y = np.zeros(D)
    # sobol_Y2 = np.zeros(D)
    tot_sobol_Y = np.zeros(D)
    # tot_sobol_Y2 = np.zeros(D)

    # Estimate Var[Y]
    mean_Y = np.mean(Y[:N_v])
    var_Y = np.var(Y[:N_v], ddof=1)
    cur_Y_idx = N_v

    # Estimate Shapley effects
    m = perms.shape[0]
    for p in range(0, m):
        cur_perm = perms[p]
        prev_C = 0
        for j in range(0, D):
            if j == D - 1:
                C_hat = var_Y
                sobol_Y[cur_perm[j]] = sobol_Y[cur_perm[j]] + prev_C  # first order effect
                # sobol_Y2[cur_perm[j]] = sobol_Y2[cur_perm[j]] + prev_C**2
            else:
                cVar = np.zeros(N_o)
                for l in range(0, N_o):
                    cVar[l] = np.var(Y[cur_Y_idx:cur_Y_idx + N_i], ddof=1)
                    cur_Y_idx += N_i

                C_hat = np.mean(cVar)

            dele = C_hat - prev_C

            shapley_Y[cur_perm[j]] = shapley_Y[cur_perm[j]] + dele
            # shapley_Y2[cur_perm[j]] = shapley_Y2[cur_perm[j]] + dele**2

            prev_C = C_hat

            if j == 0:
                tot_sobol_Y[cur_perm[j]] = tot_sobol_Y[cur_perm[j]] + C_hat  # Total effect
                # tot_sobol_Y2[cur_perm[j]] = tot_sobol_Y2[cur_perm[j]] + C_hat**2

    shapley_Y = shapley_Y / m / var_Y
    # shapley_Y2 = shapley_Y2 / m / var_Y**2
    # shapley_YSE = np.sqrt((shapley_Y2 - shapley_Y**2) / m)

    sobol_Y = sobol_Y / (m/D) / var_Y  # averaging by number of permutations with j=d-1
    # sobol_Y2 = sobol_Y2 / (m/D) / var_Y**2
    # sobol_YSE = np.sqrt((sobol_Y2 - sobol_Y**2) / (m/D))
    sobol_Y = 1 - sobol_Y
    # sobol_Y2 = 1 - sobol_Y2

    tot_sobol_Y = tot_sobol_Y / (m/D) / var_Y  # averaging by number of permutations with j=1
    # tot_sobol_Y2 = tot_sobol_Y2 / (m/D) / var_Y**2
    # tot_sobol_YSE = np.sqrt((tot_sobol_Y2 - tot_sobol_Y**2) / (m/D))

    results = dict(mu=mean_Y, sigma=var_Y, Sh=shapley_Y, S1=sobol_Y, ST=tot_sobol_Y)

    if print_to_console:
        for i in range(D):
            print("[%s] Shapley: %f | Sobol: %f | Total Sobol: %f" % (problem['names'][i],
                                                                      results['Sh'][i],
                                                                      results['S1'][i],
                                                                      results['ST'][i]))

    return results


# if __name__ == "__main__":
#     parser = common_args.create()
#     parser.add_argument('--max-order', type=int, required=False, default=2,
#                         choices=[1, 2],
#                         help='Maximum order of sensitivity indices to '
#                              'calculate')
#     parser.add_argument('-r', '--resamples', type=int, required=False,
#                         default=1000,
#                         help='Number of bootstrap resamples for Sobol '
#                              'confidence intervals')
#     parser.add_argument('--parallel', action='store_true', help='Makes '
#                         'use of parallelization.',
#                         dest='parallel')
#     parser.add_argument('--processors', type=int, required=False,
#                         default=None,
#                         help='Number of processors to be used with the ' +
#                         'parallel option.', dest='n_processors')
#     args = parser.parse_args()

#     problem = read_param_file(args.paramfile)
#     Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter,
#                    usecols=(args.column,))

#     analyze(problem, Y, (args.max_order == 2),
#             num_resamples=args.resamples, print_to_console=True,
#             parallel=args.parallel, n_processors=args.n_processors)
    
