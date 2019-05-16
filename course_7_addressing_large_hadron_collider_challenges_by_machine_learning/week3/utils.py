from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_curve
from collections import OrderedDict
import numpy
import pandas
from hep_ml import metrics


class Binner(object):
    """
    Class that helps to split the values into several bins.

    Initially an array of values is given, which is then split into
    'bins_number' equal parts,
    and thus we are computing limits (boundaries of bins).
    """

    def __init__(self, values, bins_number):
        """
        Class constructor

        Parameters
        ----------
        values : array-like
            The input distribution
        bins_number : int
            Count of bins for plot
        """

        percentiles = [i * 100.0 / bins_number for i in
                       range(1, bins_number)]
        self.limits = numpy.percentile(values, percentiles)

    def get_bins(self, values):
        """
        Given the values of feature, compute the index of bin

        Parameters
        ----------
        values : array-like, shape (n_samples,)
            The values to get the bin number from

        Returns
        -------
        np.array
            The bin numbers
        """

        return numpy.searchsorted(self.limits, values)

    def set_limits(self, limits):
        """Change the thresholds inside bins"""
        self.limits = limits

    # NOTE: Explaination of property decorator
    #       https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work
    @property
    def bins_number(self):
        """
        Returns the number of bins

        Returns
        -------
        int
            The total number of bins
        """
        return len(self.limits) + 1

    def split_into_bins(self, *arrays):
        """
        Split data into bins

        Parameters
        ----------
        arrays : argument-tuple
            Data to be split

        Returns
        -------
        results : list, shape (n_bins,)
            Values corresponding to each bin.
        """

        values = arrays[0]
        for array in arrays:
            assert len(array) == len(
                values), "passed arrays have different length"
        bins = self.get_bins(values)
        result = []
        for b in range(len(self.limits) + 1):
            indices = bins == b
            result.append(
                [numpy.array(array)[indices] for array in arrays])
        return result


def check_arrays(*arrays):
    """
    Left for consistency, a version of `sklearn.validation.check_arrays`

    Parameters
    ----------
    arrays : argument-tuple
        Input object to check / convert
        Arrays with the same length of first dimension

    Returns
    -------
    checked_arrays : object
        The converted and validated array
    """

    assert len(
        arrays) > 0, 'The number of array must be greater than zero'
    checked_arrays = []
    shapes = []
    for arr in arrays:
        if arr is not None:
            checked_arrays.append(numpy.array(arr))
            shapes.append(checked_arrays[-1].shape[0])
        else:
            checked_arrays.append(None)
    assert numpy.sum(numpy.array(shapes) == shapes[0]) == len(
        shapes), 'Different shapes of the arrays {}'.format(
        shapes)
    return checked_arrays


def check_sample_weight(y_true, sample_weight):
    """
    Asserts that the weights and predictions have the same length

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        The ground-truth
    sample_weight : None or array-like, shape (n_samples,)
        The assigned weights

    Returns
    -------
    array-like, shape (n_samples,)
        The input sample_weight if input sample_weight is not None
        An array of ones else
    """

    if sample_weight is None:
        return numpy.ones(len(y_true), dtype=numpy.float)
    else:
        sample_weight = numpy.array(sample_weight, dtype=numpy.float)
        assert len(y_true) == len(sample_weight), \
            "The length of weights is different: not {0}, but {1}".\
                format(len(y_true), len(sample_weight))
        return sample_weight


def weighted_quantile(array,
                      quantiles,
                      sample_weight=None,
                      array_sorted=False,
                      old_style=False):
    """
    Computing quantiles of an array.

    Unlike the numpy.percentile, this function supports weights,
    but it is inefficient and performs complete sorting.

    Parameters
    ----------
    array : array, shape (n_samples,)
        The input distribution
    quantiles : array-like, shape (n_quantiles,)
        Array of floats from range [0, 1] with quantiles of shape
    sample_weight : None or array-like, shape (n_samples,)
        Optional weights of the samples
    array_sorted : bool
        If True, the sorting step will be skipped
    old_style : bool
        If True, will correct output to be consistent with numpy.percentile.

    Returns
    -------
    np.array, shape (n_quantiles,)
        The values of the percentiles

    Example
    -------
    >>> weighted_quantile([1, 2, 3, 4, 5], [0.5])
    array([ 3.])
    >>> weighted_quantile([1, 2, 3, 4, 5], [0.5],
    ... sample_weight=[3, 1, 1, 1, 1])
    array([ 2.])
    """

    array = numpy.array(array)
    quantiles = numpy.array(quantiles)
    sample_weight = check_sample_weight(array, sample_weight)
    assert numpy.all(quantiles >= 0) and numpy.all(
        quantiles <= 1), 'Percentiles should be in [0, 1]'

    if not array_sorted:
        array, sample_weight = reorder_by_first(array, sample_weight)

    weighted_quantiles = numpy.cumsum(
        sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= numpy.sum(sample_weight)
    return numpy.interp(quantiles, weighted_quantiles, array)


def reorder_by_first(*arrays):
    """
    Applies the same permutation to all passed arrays.

    The order of the permutation is passed as the first array

    Parameters
    ----------
    arrays : argument-tuple
        The first element in arrays must be the order
        The arrays must have the same length of first dimension

    Returns
    -------
    list
        A list of the input arrays, ordered by the first input array
    """

    arrays = check_arrays(*arrays)
    order = numpy.argsort(arrays[0])
    return [arr[order] for arr in arrays]


def get_efficiencies(prediction,
                     spectator,
                     sample_weight=None,
                     bins_number=20,
                     thresholds=None,
                     errors=False,
                     ignored_sideband=0.0):
    """
    Constructs efficiency function dependent on spectator for each
    threshold

    Different score functions available: Efficiency, Precision, Recall,
    F1Score, and other things from sklearn.metrics

    Parameters
    ----------
    prediction : list
        List of probabilities
    spectator : list
        List of spectator's values
    sample_weight : None or array-like
        The weight given the of samples
    bins_number : int
        Count of bins for plot
    thresholds : list
        List of prediction's threshold
        (default=prediction's cuts for which efficiency will be
        [0.2, 0.4, 0.5, 0.6, 0.8])
    errors : bool
        Whether or not to include errors
    ignored_sideband : float
            A float between 0 - 1, where the number indicates the
            percent of plotting data

    Returns
    -------
    result : OrderedDict
        OrderedDict where the keys are the threshold and the values are
        tuples of arrays of the same length
        If errors is False, the values are on the form
        >>> (x_values, y_values)
        If errors is True, the values ar on the form
        >>> (x_values, y_values, y_err, x_err)
    """

    prediction, spectator, sample_weight = \
        check_arrays(prediction, spectator, sample_weight)

    spectator_min, spectator_max = \
        weighted_quantile(spectator,
                          [ignored_sideband, (1. - ignored_sideband)])
    mask = (spectator >= spectator_min) & (spectator <= spectator_max)
    spectator = spectator[mask]
    prediction = prediction[mask]
    bins_number = min(bins_number, len(prediction))
    sample_weight = sample_weight if sample_weight is None else \
        numpy.array(sample_weight)[mask]

    if thresholds is None:
        thresholds = [weighted_quantile(prediction,
                                        quantiles=1 - eff,
                                        sample_weight=sample_weight)
                      for eff in [0.2, 0.4, 0.5, 0.6, 0.8]]

    binner = Binner(spectator, bins_number=bins_number)
    if sample_weight is None:
        sample_weight = numpy.ones(len(prediction))
    bins_data = binner.split_into_bins(spectator, prediction,
                                       sample_weight)

    bin_edges = numpy.array(
        [spectator_min] + list(binner.limits) + [spectator_max])
    xerr = numpy.diff(bin_edges) / 2.
    result = OrderedDict()
    for threshold in thresholds:
        x_values = []
        y_values = []
        n_in_bin = []
        for num, (masses, probabilities, weights) in enumerate(
                bins_data):
            y_values.append(numpy.average(probabilities > threshold,
                                          weights=weights))
            n_in_bin.append(numpy.sum(weights))
            if errors:
                x_values.append(
                    (bin_edges[num + 1] + bin_edges[num]) / 2.)
            else:
                x_values.append(numpy.mean(masses))

        x_values, y_values, n_in_bin = check_arrays(x_values, y_values,
                                                    n_in_bin)
        if errors:
            result[threshold] = (x_values, y_values, numpy.sqrt(
                y_values * (1 - y_values) / n_in_bin), xerr)
        else:
            result[threshold] = (x_values, y_values)

    return result


def compute_ks(data_prediction, mc_prediction, weights_data,
               weights_mc):
    """
    Compute Kolmogorov-Smirnov (ks) distance between the real data
    predictions cdf and the Monte Carlo one.

    Parameters
    ----------
    data_prediction : array-like
        The real data predictions
    mc_prediction : array-like
        The Monte Carlo data predictions
    weights_data : array-like
        The real data weights
    weights_mc : array-like
        The Monte Carlo weights

    Returns
    -------
    Dnm : float
        The ks distance
    """

    assert len(data_prediction) == len(
        weights_data), 'Data length and weight one must be the same'
    assert len(mc_prediction) == len(
        weights_mc), 'Data length and weight one must be the same'

    data_prediction, mc_prediction = numpy.array(
        data_prediction), numpy.array(mc_prediction)
    weights_data, weights_mc = numpy.array(weights_data), numpy.array(
        weights_mc)

    assert numpy.all(data_prediction >= 0.) and numpy.all(
        data_prediction <= 1.), \
        'Data predictions are out of range [0, 1]'
    assert numpy.all(mc_prediction >= 0.) and numpy.all(
        mc_prediction <= 1.), 'MC predictions are out of range [0, 1]'

    weights_data /= numpy.sum(weights_data)
    weights_mc /= numpy.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction, mc_prediction,
                                    weights_data, weights_mc)

    dnm = numpy.max(numpy.abs(fpr - tpr))
    return dnm


def __roc_curve_splitted(data_zero,
                         data_one,
                         sample_weights_zero,
                         sample_weights_one):
    """
    Compute the roc curve with sample weights

    Parameters
    ----------
    data_zero : array-like
        Data labeled with 0
    data_one : array-like
        Data labeled with 1
    sample_weights_zero : array-like
        Weights for 0-labeled data
    sample_weights_one : array-like
        Weights for 1-labeled data

    Returns
    -------
    fpr : np.array
        The false positive rate
    tpr : np.array
        The true positive rate
    """

    labels = [0] * len(data_zero) + [1] * len(data_one)
    weights = numpy.concatenate(
        [sample_weights_zero, sample_weights_one])
    data_all = numpy.concatenate([data_zero, data_one])
    fpr, tpr, _ = roc_curve(labels, data_all, sample_weight=weights)

    return fpr, tpr


def add_noise(array, level=0.40, random_seed=34):
    """
    Adds radom noise to the input array

    Parameters
    ----------
    array : array-like
        The array to add noise to
    level : float
        The signal portion of the signal/noise ratio
    random_seed : int
        The random seed to use

    Returns
    -------
    np.array
        The original array with noise
    """

    numpy.random.seed(random_seed)

    return level * numpy.random.random(size=array.size) +\
        (1 - level) * array


# Never used in original
# NOTE: A function called `efficiencies` was also deleted from the
#       original code, see the notebook for details


def get_ks_metric(df_agree, df_test):
    """
    Returns the Kolmogorov-Smirnov (ks) metric

    Parameter
    ---------
    df_agree : DataFrame
        A dataframe containing the agreement data
    df_test : DataFrame
        A dataframe containing the test data

    Returns
    -------
    Series
        The series containing the ks distance
    """

    sig_ind = df_agree[df_agree['signal'] == 1].index
    bck_ind = df_agree[df_agree['signal'] == 0].index

    mc_prob = numpy.array(df_test.loc[sig_ind]['prediction'])
    mc_weight = numpy.array(df_agree.loc[sig_ind]['weight'])
    data_prob = numpy.array(df_test.loc[bck_ind]['prediction'])
    data_weight = numpy.array(df_agree.loc[bck_ind]['weight'])
    val, agreement_metric = check_agreement_ks_sample_weighted(
        data_prob, mc_prob, data_weight, mc_weight)

    return agreement_metric['ks']


def check_agreement_ks_sample_weighted(data_prediction,
                                       mc_prediction,
                                       weights_data,
                                       weights_mc):
    """
    Checks the agreement between the data prediction and the Monte
    Carlo prediction

    Parameters
    ----------
    data_prediction : array-like
        Predictions from the data
    mc_prediction : array-like
        Predictions from the Monte Carlo simulations
    weights_data : array-like
        Weights for the real data
    weights_mc : array-like
        Wight for the Monte Carlo simulation

    Returns
    -------
    bool
        Whether or not the ks distance part is less than 0.03
    result : Dict
        Dictionary on the form
        >>> {'ks': ks_distance, 'ks_part': ks_distance_part}
    """

    data_prediction, weights_data = map(column_or_1d,
                                        [data_prediction, weights_data])
    mc_prediction, weights_mc = map(column_or_1d,
                                    [mc_prediction, weights_mc])

    assert numpy.all(data_prediction >= 0.) and numpy.all(
        data_prediction <= 1.), 'error in prediction'
    assert numpy.all(mc_prediction >= 0.) and numpy.all(
        mc_prediction <= 1.), 'error in prediction'

    weights_data = weights_data / numpy.sum(weights_data)
    weights_mc = weights_mc / numpy.sum(weights_mc)

    data_neg = data_prediction[weights_data < 0]
    weights_neg = -weights_data[weights_data < 0]
    mc_prediction = numpy.concatenate((mc_prediction, data_neg))
    weights_mc = numpy.concatenate((weights_mc, weights_neg))
    data_prediction = data_prediction[weights_data >= 0]
    weights_data = weights_data[weights_data >= 0]

    assert numpy.all(weights_data >= 0) and numpy.all(weights_mc >= 0)
    assert numpy.allclose(weights_data.sum(), weights_mc.sum())

    weights_data /= numpy.sum(weights_data)
    weights_mc /= numpy.sum(weights_mc)

    fpr, tpr = __roc_curve_splitted(data_prediction,
                                    mc_prediction,
                                    weights_data,
                                    weights_mc)

    dnm = numpy.max(numpy.abs(fpr - tpr))
    dnm_part = numpy.max(numpy.abs(fpr - tpr)[fpr + tpr < 1])

    result = {'ks': dnm, 'ks_part': dnm_part}
    return dnm_part < 0.03, result


def check_correlation(probabilities, mass):
    """
    Checks the correlation between probabilities and mass

    Parameters
    ----------
    probabilities : array-like
        Array of the probabilities
    mass : array-like
        Array of the corresponding masses

    Returns
    -------
    np.array
        The Cramer-von Mises distance between the two distributions
    """
    probabilities, mass = map(column_or_1d, [probabilities, mass])

    y_pred = numpy.zeros(shape=(len(probabilities), 2))
    y_pred[:, 1] = probabilities
    y_pred[:, 0] = 1 - probabilities
    y_true = [0] * len(probabilities)
    df_mass = pandas.DataFrame({'mass': mass})
    cvm = metrics.BinBasedCvM(uniform_features=['mass'],
                              uniform_label=0)
    cvm.fit(df_mass, y_true)
    return cvm(y_true, y_pred, sample_weight=None)
