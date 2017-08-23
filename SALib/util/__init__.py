"""A set of utility functions

"""
from collections import OrderedDict
import csv
from warnings import warn

import numpy as np
import scipy as sp

__all__ = ["scale_samples", "read_param_file"]


def scale_samples(params, bounds):
    '''Rescale samples in 0-to-1 range to arbitrary bounds

    Arguments
    ---------
    bounds : list
        list of lists of dimensions `num_params`-by-2
    params : numpy.ndarray
        numpy array of dimensions `num_params`-by-:math:`N`,
        where :math:`N` is the number of samples
    '''
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(params,
                       (upper_bounds - lower_bounds),
                       out=params),
           lower_bounds,
           out=params)


def unscale_samples(params, bounds):
    """Rescale samples from arbitrary bounds back to [0,1] range

    Arguments
    ---------
    bounds : list
        list of lists of dimensions num_params-by-2
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    """
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   (sample - lower_bound) / (upper_bound - lower_bound)
    np.divide(np.subtract(params, lower_bounds, out=params),
              np.subtract(upper_bounds, lower_bounds),
              out=params)


def nonuniform_scale_samples(params, bounds, dists):
    """Rescale samples in 0-to-1 range to other distributions

    Arguments
    ---------
    problem : dict
        problem definition including bounds
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    dists : list
        list of distributions, one for each parameter
            unif: uniform with lower and upper bounds
            triang: triangular with width (scale) and location of peak
                    location of peak is in percentage of width
                    lower bound assumed to be zero
            norm: normal distribution with mean and standard deviation
            lognorm: lognormal with ln-space mean and standard deviation
            isotriang: isosceles triangular distribution
                       lower and upper bounds  (peak = 0.5%)
            gumbel: gumbel distribution with mode and scale
            weibull: weibull distribution with shape and scale
    """
    b = np.array(bounds)

    # initializing matrix for converted values
    conv_params = np.zeros_like(params)

    # loop over the parameters
    for i in range(conv_params.shape[1]):
        # setting first and second arguments for distributions
        b1 = b[i][0]
        b2 = b[i][1]
        hard_b1 = b[i][2] if len(b[i]) > 2 else np.NAN
        hard_b2 = b[i][3] if len(b[i]) > 3 else np.NAN
        lower = 0
        upper = 1

        if dists[i] == 'triang':
            # checking for correct parameters
            if b1 <= 0 or b2 <= 0 or b2 >= 1:
                raise ValueError('''Triangular distribution: Scale must be
                    greater than zero; peak on interval [0,1]''')
            else:
                conv_params[:, i] = sp.stats.triang.ppf(
                    params[:, i], c=b2, scale=b1, loc=0)

        elif dists[i] == 'unif':
            if b1 >= b2:
                raise ValueError('''Uniform distribution: lower bound
                    must be less than upper bound''')
            else:
                conv_params[:, i] = params[:, i] * (b2 - b1) + b1

        elif dists[i] == 'norm':
            if b2 <= 0:
                raise ValueError('''Normal distribution: stdev must be > 0''')
            else:
                if hard_b1 != np.NAN:
                    lower = sp.stats.norm.cdf(hard_b1, loc=b1, scale=b2)
                if hard_b2 != np.NAN:
                    upper = sp.stats.norm.cdf(hard_b2, loc=b1, scale=b2)
                conv_params[:, i] = sp.stats.norm.ppf(
                    params[:, i] * (upper - lower) + lower,
                    loc=b1, scale=b2)

        # lognormal distribution (ln-space, not base-10)
        # parameters are ln-space mean and standard deviation
        elif dists[i] == 'lognorm':
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError(
                    '''Lognormal distribution: stdev must be > 0''')
            else:
                if hard_b1 != np.NAN:
                    lower = sp.stats.norm.cdf(hard_b1, loc=b1, scale=b2)
                if hard_b2 != np.NAN:
                    upper = sp.stats.norm.cdf(hard_b2, loc=b1, scale=b2)
                conv_params[:, i] = np.exp(
                    sp.stats.norm.ppf(params[:, i], loc=b1, scale=b2))

        # isosceles triangular distribution
        # parameters are the location of the lower and upper bounds (peak=0.5%)
        elif dists[i] == 'isotriang':
            # checking for valid parameters
            if b1 >= b2:
                raise ValueError('''Isosceles triangular distribution: lower
                    bound must be less than upper bound''')
            else:
                conv_params[:, i] = sp.stats.triang.ppf(
                    params[:, i], c=0.5, scale=b2 - b1, loc=b1)

        # gumbel distribution
        # parameters are the location of the mode and the scale
        elif dists[i] == 'gumbel':
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError(
                    '''Gumbel distribution: scale must be > 0''')
            else:
                if hard_b1 != np.NAN:
                    lower = sp.stats.gumbel_r.cdf(hard_b1, loc=b1, scale=b2)
                if hard_b2 != np.NAN:
                    upper = sp.stats.gumbel_r.cdf(hard_b2, loc=b1, scale=b2)
                conv_params[:, i] = sp.stats.gumbel_r.ppf(
                    params[:, i] * (upper - lower) + lower,
                    loc=b1, scale=b2)

        # weibull distribution
        # parameters are the shape and the scale of the distribution
        elif dists[i] == 'weibull':
            # checking for valid parameters
            if b1 <= 0:
                raise ValueError(
                    '''Weibull distribution: shape must be > 0''')
            elif b2 <= 0:
                raise ValueError(
                    '''Weibull distribution: scale must be > 0''')
            else:
                if hard_b1 != np.NAN:
                    lower = sp.stats.weibull_min.cdf(hard_b1, c=b1, scale=b2)
                if hard_b2 != np.NAN:
                    upper = sp.stats.weibull_min.cdf(hard_b2, c=b1, scale=b2)
                conv_params[:, i] = sp.stats.weibull_min.ppf(
                    params[:, i] * (upper - lower) + lower,
                    c=b1, scale=b2)

        else:
            valid_dists = ['unif', 'triang', 'norm', 'lognorm',
                           'isotriang', 'gumbel', 'weibull']
            raise ValueError('Distributions: choose one of %s' %
                             ", ".join(valid_dists))

    return conv_params


def read_param_file(filename, delimiter=None):
    """Unpacks a parameter file into a dictionary

    Reads a parameter file of format::

        Param1,0,1,Group1,dist1
        Param2,0,1,Group2,dist2
        Param3,0,1,Group3,dist3

    (Group and Dist columns are optional)

    Returns a dictionary containing:
        - names - the names of the parameters
        - bounds - a list of lists of lower and upper bounds
        - num_vars - a scalar indicating the number of variables
                     (the length of names)
        - groups - a tuple containing i) a group matrix assigning parameters to
                   groups
                                      ii) a list of unique group names
        - dists - a list of distributions for the problem,
                    None if not specified or all uniform

    Arguments
    ---------
    filename : str
        The path to the parameter file
    delimiter : str, default=None
        The delimiter used in the file to distinguish between columns

    """
    names = []
    bounds = []
    groups = []
    dists = []
    num_vars = 0
    fieldnames = ['name', 'lower_bound', 'upper_bound', 'group', 'dist']

    with open(filename, 'rU') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=delimiter)
        csvfile.seek(0)
        reader = csv.DictReader(
            csvfile, fieldnames=fieldnames, dialect=dialect)
        for row in reader:
            if row['name'].strip().startswith('#'):
                pass
            else:
                num_vars += 1
                names.append(row['name'])
                bounds.append(
                    [float(row['lower_bound']), float(row['upper_bound'])])

                # If the fourth column does not contain a group name, use
                # the parameter name
                if row['group'] is None:
                    groups.append(row['name'])
                elif row['group'] is 'NA':
                    groups.append(row['name'])
                else:
                    groups.append(row['group'])

                # If the fifth column does not contain a distribution
                # use uniform
                if row['dist'] is None:
                    dists.append('unif')
                else:
                    dists.append(row['dist'])

    # group_matrix, group_names = compute_groups_from_parameter_file(
    #     groups, num_vars)

    if groups == names:
        groups = None
    elif len(set(groups)) == 1:
        raise ValueError('''Only one group defined, results will not be
            meaningful''')

    # setting dists to none if all are uniform
    # because non-uniform scaling is not needed
    if all([d == 'unif' for d in dists]):
        dists = None

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars,
            'groups': groups, 'dists': dists}


def compute_groups_matrix(groups, num_vars):
    """Generate matrix which notes factor membership of groups

    Computes a k-by-g matrix which notes factor membership of groups
    where:
        k is the number of variables (factors)
        g is the number of groups
    Also returns a g-length list of unique group_names whose positions
    correspond to the order of groups in the k-by-g matrix

    Returns
    -------
    tuple
        containing group matrix assigning parameters to
        groups and a list of unique group names
    """
    if not groups:
        return None

    # Get a unique set of the group names
    unique_group_names = list(OrderedDict.fromkeys(groups))
    number_of_groups = len(unique_group_names)

    indices = dict([(x, i) for (i, x) in enumerate(unique_group_names)])

    output = np.zeros((num_vars, number_of_groups), dtype=np.int)

    for parameter_row, group_membership in enumerate(groups):
        group_index = indices[group_membership]
        output[parameter_row, group_index] = 1

    return np.matrix(output), unique_group_names


def requires_gurobipy(_has_gurobi):
    '''
    Decorator function which takes a boolean _has_gurobi as an argument.
    Use decorate any functions which require gurobi.
    Raises an import error at runtime if gurobi is not present.
    Note that all runtime errors should be avoided in the working code,
    using brute force options as preference.
    '''
    def _outer_wrapper(wrapped_function):
        def _wrapper(*args, **kwargs):
            if _has_gurobi:
                result = wrapped_function(*args, **kwargs)
            else:
                warn("Gurobi not available", ImportWarning)
                result = None
            return result
        return _wrapper
    return _outer_wrapper
