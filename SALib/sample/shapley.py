from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import scipy as sp

# from . import common_args
# from ..util import scale_samples, nonuniform_scale_samples, read_param_file, compute_groups_matrix

#
#   Source code adapted from R 'sensitivity' package:
#       https://github.com/cran/sensitivity/blob/master/R/shapleyPermEx.R
#       https://github.com/cran/sensitivity/blob/master/R/shapleyPermRand.R
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


def sample_ex(problem, N_v, N_o, N_i,
              joint_rand_fn, cond_rand_fn):
    """Generates model inputs for computation of Shapley effects.

    Returns a NumPy matrix containing the model inputs. The resulting matrix
    has N_v + N_o * N_i * D * D! rows, where D is the number of parameters.
    These model inputs are intended to be used with :func:`SALib.analyze.shapley.analyze_ex`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N_v : int
        Monte Carlo (MC) sample size to estimate the output variance
    N_o : int
        Output MC sample size to estimate the cost function
    N_i : int
        Inner MC sample size to estimate the cost function
    joint_rand_fn : lambda n, d: numpy.matrix([n, d])
        A function to generate a matrix with n samples (in rows) from the
        D-dimensional input joint PDF
    cond_rand_fn : lambda n, dep_idx_set, given_idx_set, given_values: numpy.matrix([n, len(dep_idx_set)])
        A function to generate a matrix with n samples (in rows) from the
        len(dep_idx_set)-dimensional conditional PDF when given_idx_set take given_values

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
    """

    assert problem.get('groups', None) is None

    D = problem['num_vars']
    perms = np.asarray(list(itertools.permutations(range(D))))
    m = perms.shape[0]
    X = np.empty([N_v + m * (D - 1) * N_o * N_i, D])
    X[:N_v, :] = joint_rand_fn(N_v)

    for p in range(m):
        cur_perm = perms[p]

        for j in range(1, D):
            S_j = cur_perm[:j]  # set of the 1st-jth elements in cur_perm
            cS_j = cur_perm[j:]  # set of the (j+1)th-dth elements in cur_perm
            cMx_j = joint_rand_fn(N_o, columns_set=cS_j)  # sampled values of the inputs in cS_j

            idx = N_v + N_o * N_i * (p * (D - 1) + (j - 1))
            for l in range(N_o):
                cx_j = cMx_j[l]
                # sample values of inputs in S_j conditional on cx_j
                X[idx + l * N_i:idx + (l+1) * N_i, :] = cond_rand_fn(N_i, S_j, cS_j, cx_j)

    return X


def sample_rand(problem, m, N_v, N_o, N_i,
                joint_rand_fn, cond_rand_fn):
    """Generates model inputs for computation of Shapley effects.

    Returns a NumPy matrix containing the model inputs. The resulting matrix
    has N_v + N_o * N_i * D * D! rows, where D is the number of parameters.
    These model inputs are intended to be used with :func:`SALib.analyze.shapley.analyze_rand`.

    Parameters
    ----------
    problem : dict
        The problem definition
    m : int
        Number of randomly sampled permutations
    N_v : int
        Monte Carlo (MC) sample size to estimate the output variance
    N_o : int
        Output MC sample size to estimate the cost function
    N_i : int
        Inner MC sample size to estimate the cost function
    joint_rand_fn : lambda n, d: numpy.matrix([n, d])
        A function to generate a matrix with n samples (in rows) from the
        D-dimensional input joint PDF
    cond_rand_fn : lambda n, dep_idx_set, given_idx_set, given_values: numpy.matrix([n, len(dep_idx_set)])
        A function to generate a matrix with n samples (in rows) from the
        len(dep_idx_set)-dimensional conditional PDF when given_idx_set take given_values

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
    """

    assert problem.get('groups', None) is None
    D = problem['num_vars']
    perms = np.empty((m, D), dtype=np.int64)
    dims = np.arange(D)
    for i in range(m):
        perms[i] = np.random.permutation(dims)

    X = np.empty([N_v + m * (D - 1) * N_o * N_i, D])
    X[:N_v, :] = joint_rand_fn(N_v)

    print("Generating random queries...")
    for p in range(m):
        cur_perm = perms[p]
        if p % ( m // 100) == 0:
            print("\r{}% done".format(p // ( m // 100)), end='')
        for j in range(1, D):
            S_j = cur_perm[:j]  # set of the 1st-jth elements in cur_perm
            cS_j = cur_perm[j:]  # set of the (j+1)th-dth elements in cur_perm
            cMx_j = joint_rand_fn(N_o, columns_set=cS_j)  # sampled values of the inputs in cS_j

            idx = N_v + N_o * N_i * (p * (D - 1) + (j - 1))
            for l in range(N_o):
                cx_j = cMx_j[l]
                # sample values of inputs in S_j conditional on cx_j
                X[idx + l * N_i:idx + (l+1) * N_i, :] = cond_rand_fn(N_i, S_j, cS_j, cx_j)

    return X, perms


# if __name__ == "__main__":

#     parser = common_args.create()

#     parser.add_argument(
#         '-n', '--samples', type=int, required=True, help='Number of Samples')

#     parser.add_argument('--max-order', type=int, required=False, default=2,
#                         choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
#     args = parser.parse_args()

#     np.random.seed(args.seed)
#     problem = read_param_file(args.paramfile)

#     param_values = sample(problem, args.samples, calc_second_order=(args.max_order == 2))
#     np.savetxt(args.output, param_values, delimiter=args.delimiter,
#                fmt='%.' + str(args.precision) + 'e')
