"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
import warnings


import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
#from pyDOE2 import lhs
from scipy.stats.distributions import norm

'''Changed from lime.discretize to lime.nandiscretize (just another copy of the file)'''
'''
from lime.discretize import QuartileDiscretizer
from lime.nandiscretize import DecileDiscretizer
from lime.nandiscretize import EntropyDiscretizer
from lime.nandiscretize import BaseDiscretizer
from lime.nandiscretize import StatsDiscretizer
'''
#from . import explanation
import explanation
#from . import lime_base
import lime_base

'''from .nandiscretize import BaseDiscretizer
from .nandiscretize import QuartileDiscretizer
from .nandiscretize import QuartileDiscretizerMV
from .nandiscretize import DecileDiscretizer
from .nandiscretize import EntropyDiscretizer
from .nandiscretize import StatsDiscretizer'''

from nandiscretize import BaseDiscretizer
from nandiscretize import QuartileDiscretizer
from nandiscretize import QuartileDiscretizerMV
from nandiscretize import DecileDiscretizer
from nandiscretize import EntropyDiscretizer
from nandiscretize import StatsDiscretizer

from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer

import math


class TableDomainMapper(explanation.DomainMapper):
    """Maps feature ids to names, generates table views, etc"""

    def __init__(self, feature_names, feature_values, scaled_row,
                 categorical_features, discretized_feature_names=None,
                 feature_indexes=None):
        """Init.

        Args:
            feature_names: list of feature names, in order
            feature_values: list of strings with the values of the original row
            scaled_row: scaled row
            categorical_features: list of categorical features ids (ints)
            feature_indexes: optional feature indexes used in the sparse case
        """
        self.exp_feature_names = feature_names
        self.discretized_feature_names = discretized_feature_names
        self.feature_names = feature_names
        self.feature_values = feature_values
        self.feature_indexes = feature_indexes
        self.scaled_row = scaled_row
        if sp.sparse.issparse(scaled_row):
            self.all_categorical = False
        else:
            self.all_categorical = len(categorical_features) == len(scaled_row)
        self.categorical_features = categorical_features

    def map_exp_ids(self, exp):
        """Maps ids to feature names.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight)
        """
        names = self.exp_feature_names
        if self.discretized_feature_names is not None:
            names = self.discretized_feature_names
        return [(names[x[0]], x[1]) for x in exp]

    def visualize_instance_html(self,
                                exp,
                                label,
                                div_name,
                                exp_object_name,
                                show_table=True,
                                show_all=False):
        """Shows the current example in a table format.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             show_table: if False, don't show table visualization.
             show_all: if True, show zero-weighted features in the table.
        """
        if not show_table:
            return ''
        weights = [0] * len(self.feature_names)
        for x in exp:
            weights[x[0]] = x[1]
        if self.feature_indexes is not None:
            # Sparse case: only display the non-zero values and importances
            fnames = [self.exp_feature_names[i] for i in self.feature_indexes]
            fweights = [weights[i] for i in self.feature_indexes]
            if show_all:
                out_list = list(zip(fnames,
                                    self.feature_values,
                                    fweights))
            else:
                out_dict = dict(map(lambda x: (x[0], (x[1], x[2], x[3])),
                                zip(self.feature_indexes,
                                    fnames,
                                    self.feature_values,
                                    fweights)))
                out_list = [out_dict.get(x[0], (str(x[0]), 0.0, 0.0)) for x in exp]
        else:
            out_list = list(zip(self.exp_feature_names,
                                self.feature_values,
                                weights))
            if not show_all:
                out_list = [out_list[x[0]] for x in exp]
        ret = u'''
            %s.show_raw_tabular(%s, %d, %s);
        ''' % (exp_object_name, json.dumps(out_list, ensure_ascii=False), label, div_name)
        return ret


class LimeTabularExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',#'auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):
        """Init function.

        Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
                If None, defaults to sqrt (number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True
                and data is not sparse. Options are 'quartile', 'decile',
                'entropy' or a BaseDiscretizer instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
            training_data_stats: a dict object having the details of training data
                statistics. If None, training data information will be used, only matters
                if discretize_continuous is True. Must have the following keys:
                means", "mins", "maxs", "stds", "feature_values",
                "feature_frequencies"
        """
        self.random_state = check_random_state(random_state)
        self.mode = mode
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        self.training_data_stats = training_data_stats
        #self.newcheck = []


        # Check and raise proper error in stats are supplied in non-descritized path
        if self.training_data_stats:
            self.validate_training_data_stats(self.training_data_stats)

        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                discretizer = StatsDiscretizer(
                    training_data, self.categorical_features,
                    self.feature_names, labels=training_labels,
                    data_stats=self.training_data_stats,
                    random_state=self.random_state)

            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'quartileMV':
                self.discretizer = QuartileDiscretizerMV(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels,
                        random_state=self.random_state)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))

            # Get the discretized_training_data when the stats are not provided
            if(self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(
                    training_data)

        if kernel_width is None:
            kernel_width = np.sqrt(training_data.shape[1]) * .75
        kernel_width = float(kernel_width)

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.feature_selection = feature_selection
        self.base = lime_base.LimeBase(kernel_fn, verbose, random_state=self.random_state)
        self.class_names = class_names



        # Though set has no role to play if training data stats are provided
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            if training_data_stats is None:
                if self.discretizer is not None:
                    column = discretized_training_data[:, feature]
                else:
                    column = training_data[:, feature]

                feature_count = collections.Counter(column)
                values, frequencies = map(list, zip(*(sorted(feature_count.items()))))
            else:
                values = training_data_stats["feature_values"][feature]
                frequencies = training_data_stats["feature_frequencies"][feature]

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

    @staticmethod
    def validate_training_data_stats(training_data_stats):
        """
            Method to validate the structure of training data stats
        """
        stat_keys = list(training_data_stats.keys())
        valid_stat_keys = ["means", "mins", "maxs", "stds", "feature_values", "feature_frequencies"]
        missing_keys = list(set(valid_stat_keys) - set(stat_keys))
        if len(missing_keys) > 0:
            raise Exception("Missing keys in training_data_stats. Details: %s" % (missing_keys))

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian'):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. For classifiers, this should be a
                function that takes a numpy array and outputs prediction
                probabilities. For regressors, this takes a numpy array and
                returns the predictions. For ScikitClassifiers, this is
                `classifier.predict_proba()`. For ScikitRegressors, this
                is `regressor.predict()`. The prediction function needs to work
                on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have model_regressor.coef_
                and 'sample_weight' as a parameter to model_regressor.fit()
            sampling_method: Method to sample synthetic data. Defaults to Gaussian
                sampling. Can also use Latin Hypercube Sampling.

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """


        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        data, inverse = self.__data_inverse(data_row, num_samples, sampling_method)
        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def __data_inverse(self,
                       data_row,
                       num_samples,
                       sampling_method):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols) #cat features ranges from 0 to all the columns. WHY??? -RISOLTO-
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == 'gaussian':
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1]*num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features #which would be [] in this case.
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row) #Input row, Discretized
        data[0] = data_row.copy() #Data is the matrix of generated sample. Use the input row as first element.
        inverse = data.copy() #Create a copy of data, which is the input row and 4999 rows of zeroes.
        for column in categorical_features: #After discretizing, all features become categorical!!
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse

############################################################################################################################################

class LimeTabularExplainerMV(LimeTabularExplainer):
    """
    A modification of LimeTabularExplainer capable of handling the presence of missing values.
    The neighborhood generation procedure creates synthetic data where missing values are present.
    (see __data_inverse)
    Then we use a model capable of working on incomplete data instead of a linear regressor
    model (see lime_base.py, where we need to create a lime_baseMV.)
    """
    def __init__(self,
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='none', #auto
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None,
                 sample_removal = None, #NEW -->
                 surrogate_data = 'continuous', 
                 custom_sample = None, 
                 tree_explainer = False,
                 distance_type = 'binary', 
                 to_round = None):

        '''super().__init__(
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=feature_names,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='none', #auto
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None)'''

        super().__init__(
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=feature_names, #CAMBIATO
                 categorical_features=categorical_features, #CAMBIATO
                 categorical_names=categorical_names, #CAMBIATO DA NONE
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=class_names, #CAMBIATO da NONE
                 feature_selection='none', #CAMBIATO DA 'none'
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None),
                 #to_round=to_round)

        ######### NEW THINGS ADDED
        #Check completeness and compute statistics
        self.local_proba = []
        self.surrogate_data = surrogate_data
        self.custom_sample = custom_sample
        self.train_complete = True
        self.current_test_complete = True
        self.tree_explainer = tree_explainer
        self.distance_type = distance_type
        passed_data = []
        for feature in self.categorical_features:
            feature_summary = {}
            if feature_names is not None:
                feature_summary['name'] = feature_names.copy()[feature]
            else:
                feature_summary['name'] = 'Feature '+str(feature)
            feature_summary['mean'] = np.nanmean(training_data[:,feature])
            feature_summary['std'] = np.nanstd(training_data[:,feature])
            feature_summary['min'] = np.nanmin(training_data[:,feature])
            feature_summary['max'] = np.nanmax(training_data[:,feature])
            nan_count = np.count_nonzero(np.isnan(training_data[:,feature]))
            if nan_count != 0:
                self.train_complete=False
            feature_summary['mv%'] = round(nan_count/training_data.shape[0], 3)
            passed_data.append(feature_summary)
        self.passed_data = passed_data

        self.feature_names = feature_names 
        self.check_explain = {}
        self.to_round = to_round

        #IMPLEMENT REMOVAL OF VALUES FROM THE SAMPLE.
        self.removal = [0] * len(self.categorical_features)
        if sample_removal is not None:
            if sample_removal == 'auto':
                for i, feature_info in enumerate(self.passed_data):
                    self.removal[i] = feature_info['mv%']
            elif (type(sample_removal) is list) and (len(sample_removal) == len(self.categorical_features)):
                if all((x <= 1) and (x >= 0) for x in sample_removal):
                    self.removal = sample_removal
                else:
                    raise Exception("elements of sample_removal have to be between 0 and 1, included")
            else:
                raise Exception("sample_removal can be None, 'auto', or a list containing a percentage for each feature in the data")

        self.checkneighdata = [] #shows the data used as synthetic neighborhood.
        self.checkyss = [] #shows the prediction probabilities for each class, computed on the synthetic neighborhood.
        self.checkdistance = [] #shows the data used to compute the distances from synthetic records to the original test record.

        ##New: IMPUTER (for KNN only)
        if self.custom_sample == 'knn':
            self.imputer = KNNImputer(n_neighbors=2, missing_values=np.nan)
            self.imputer.fit(training_data)

        if self.custom_sample=='mice':
            self.imputer = IterativeImputer(random_state=42)
            self.imputer.fit(training_data)
        ######### 
        
    def explain_instance(self,
                         data_row,
                         predict_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian'):#, 
                         #tree_explainer = False, 
                         #distance_type = 'binary'):#,
                         #custom_nan_test = True, #Use one of our methods for handling NANs in the test row
                         #sample_imputer=None,
                         #remove_percentage = 'auto' #auto for train MV distribution, list of [0,1] for manual
                         #): ##new
       
        ##### NEW THINGS 
        # --> Check test completeness
        self.current_test_complete = True
        if np.count_nonzero(np.isnan(data_row)) != 0:
            self.current_test_complete = False 

        #Check that model regressor is not None if self.removal has non-zero values
        if model_regressor is None and any(x != 0 for x in self.removal):
            warnings.warn("Since sample_removal was not none, setting model_regressor to 'tree' instead of None.")
            model_regressor = 'tree'

        ##### 



        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        
        data, inverse = self.__data_inverse(data_row, num_samples, sampling_method)
 
        if self.surrogate_data == 'continuous':

            if self.distance_type != 'continuous': 
                #make a copy to account for the case where we want to use distances on the binary data, but train regressor on the continuous data.
                binary_data = np.copy(data)
                if sp.sparse.issparse(data):
                    binary_data = binary_data.multiply(self.scaler.scale_)
                    if not sp.sparse.isspmatrix_csr(binary_data):
                        binary_data = binary_data.tocsr()
                else:
                    binary_data = (binary_data - self.scaler.mean_) / self.scaler.scale_

                    #print('BINARYDATA \n', binary_data)

            data = np.copy(inverse) 
            #print('DATA \n', data)

        elif self.distance_type == 'continuous':
            warnings.warn("Since distance_type is set to continuous, this continuous data will be also passed to the regressor model.")
            self.surrogate_data = 'continuous'
            



       

             
        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        #print('scaled data', scaled_data)

        #Handles the presence of missing values in the sample neighborhood, and computes distances.

        if np.isnan(scaled_data).any():
            if distance_metric != 'euclidean':
                warnings.warn("Since 'remove_sample' is not None, the generated sample contains NaNs. As such, the distance metric used to compute weights has been set to 'euclidean'.")
            
            #need to do this in BOTH continuous and binary data in case they're used together. Modifications should pass to data_for_distances.
            scaled_data = np.delete(scaled_data, np.argwhere(np.isnan(scaled_data).all(axis=1)), axis=0)
            #print('scaled_data', scaled_data.shape)
            if 'binary_data' in locals():
                binary_data = np.delete(binary_data, np.argwhere(np.isnan(binary_data).all(axis=1)), axis=0)
                #print('binary_data', binary_data.shape)

            data_for_distances = scaled_data
            if self.distance_type != 'continuous' and self.surrogate_data == 'continuous':
                data_for_distances = binary_data #use the copy

            #print('before inverse', inverse.shape)
            inverse = np.delete(inverse, np.argwhere(np.isnan(inverse).all(axis=1)), axis=0)
            #print('after inverse', inverse.shape)
            distances = sklearn.metrics.pairwise.nan_euclidean_distances(
                data_for_distances,
                data_for_distances[0].reshape(1, -1),
        ).ravel()
        else:
            data_for_distances = scaled_data
            if self.distance_type != 'continuous' and self.surrogate_data == 'continuous':
                data_for_distances = binary_data #use the copy

            distances = sklearn.metrics.pairwise_distances(
                    data_for_distances,
                    data_for_distances[0].reshape(1, -1),
                    metric=distance_metric,
                    force_all_finite = False #new, does not do anything though.
            ).ravel()
        self.checkdistance.append(data_for_distances) #after
        #print('distances', distances.shape)


        if np.isnan(data_row).any():
            idx = np.argwhere(np.isnan(distances))
            #print('idx \n', idx.shape)
            scaled_data = np.delete(scaled_data, idx, axis=0)
            inverse = np.delete(inverse, idx, axis=0)
            #print('scaled data shape \n', scaled_data.shape)
            if 'binary_data' in locals():
                binary_data = np.delete(binary_data, idx, axis=0)
                #print('binary data shape \n', binary_data.shape)
            distances = np.delete(distances, idx, axis=0) 
            #print('distances new shape \n', distances.shape)

        yss = predict_fn(inverse) 
        #print('yss', yss.shape)
        self.checkyss = yss

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            if np.isnan(data_row[i]):
                name = 'Missing'
            else:
                name = int(data_row[i])
            if i in self.categorical_names and name!= 'Missing':
                name = self.categorical_names[i][name]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features

        discretized_feature_names = None
        
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)

            '''self.newcheck.append(self.discretizer.names)
            print(self.discretizer.names)
            print(discretizer_feature_names)
            self.newcheck.append(discretizer_feature_names)'''
            self.check_explain['self.discretizer.names'] = self.discretizer.names
            '''print('!!!!')
            print('self.check_explain', self.check_explain)'''
            self.check_explain['discretized_feature_names'] = discretized_feature_names
            self.check_explain['discretized_instance'] = discretized_instance

            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        for i, dfn in enumerate(discretized_feature_names):
            if values[i] == 'nan':
                discretized_feature_names[i] = '%s = NaN' %(feature_names[i])

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]

        self.checkneighdata = scaled_data
        
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score[label],
             ret_exp.local_pred[label]) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection,
                    tree_explainer = self.tree_explainer)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def __data_inverse(self,
                       data_row,
                       num_samples,
                       sampling_method):#,
                       #custom_nan_test,
                       #imp):
        """Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Args:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model
            sampling_method: 'gaussian' or 'lhs'

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == 'gaussian':
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1]*num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        self.check=[]
        col_nan_idx = [] #list of columns not to undiscretize.
        for column in categorical_features:
            if (math.isnan(data_row[column])):
                if self.custom_sample in ['mean', 'uniform', 'normal']:
                    col_nan_idx.append(column)
                    inverse_column = np.empty((num_samples))
                    inverse_column[:] = np.nan
                    if self.custom_sample == 'mean':
                        #print('mean', column)
                        inverse_column[:] = self.passed_data[column]['mean']
                        #If "filling" with mean, all the rows have the same value, so all 1 in data.
                        binary_column = np.empty((num_samples))
                        binary_column[:] = 1
                    elif self.custom_sample == 'uniform':
                        #print('uniform', column)
                        inverse_column[:] = np.random.uniform(self.passed_data[column]['min'], self.passed_data[column]['max'], size=num_samples)
                        #If filling with a random method, the feature is perturbed in any case, so all 0 in data (and 1 in the first row) 
                        binary_column = np.empty((num_samples))
                        binary_column[:] = 0
                        binary_column[0] = 1
                    elif self.custom_sample == 'normal':
                        #print('normal', column)
                        inverse_column[:] = np.random.uniform(self.passed_data[column]['mean'], self.passed_data[column]['std'], size=num_samples)
                        #If filling with a random method, the feature is perturbed in any case, so all 0 in data (and 1 in the first row) 
                        binary_column = np.empty((num_samples))
                        binary_column[:] = 0
                        binary_column[0] = 1

                elif self.custom_sample in [None, 'knn', 'mice']:
                    #default behaviour. We do this for KNN as well, since we need the inverse for the other columns before applying it.
                    values = self.feature_values[column]
                    freqs = self.feature_frequencies[column]
                    inverse_column = self.random_state.choice(values, size=num_samples,
                                                            replace=True, p=freqs)
                    binary_column = (inverse_column == first_row[column]).astype(int)
                    binary_column[0] = 1
                    inverse_column[0] = data[0, column]
                    data[:, column] = binary_column
                else:
                    raise Exception("The test row has missing values. Please choose custom_sample from None, 'mean', 'knn', 'uniform' or 'normal'")
                    #Don't raise exception if the value is wrong but the test column is complete.
            
            else:
                #default behaviour
                values = self.feature_values[column]
                freqs = self.feature_frequencies[column]
                inverse_column = self.random_state.choice(values, size=num_samples,
                                                        replace=True, p=freqs)
                binary_column = (inverse_column == first_row[column]).astype(int)
                binary_column[0] = 1
                inverse_column[0] = data[0, column]
                data[:, column] = binary_column

            inverse[:, column] = inverse_column
        ###########################################

        if self.discretizer is not None:    
            
            to_discretize = [x for x in categorical_features if x not in col_nan_idx]      
            inverse[1:, :] = self.discretizer.selective_undiscretize(data=inverse[1:, :], idx=to_discretize) 
        inverse[0] = data_row

        ###########################################
        #imputation with KNN (for random, need to specify here what parameters to use)
        if self.custom_sample == 'knn':
            for column in categorical_features:
                if (math.isnan(data_row[column])):
                    inverse_empty_column = np.empty((num_samples))
                    inverse_empty_column[:] = np.nan
                    inverse[:, column]=inverse_empty_column
            ###Imputation on inverse
            inverse[1:] = self.imputer.transform(inverse[1:])
        #self.check = (data, inverse, self.removal) #CHECK

        if self.custom_sample == 'mice':
            for column in categorical_features:
                if (math.isnan(data_row[column])):
                    inverse_empty_column = np.empty((num_samples))
                    inverse_empty_column[:] = np.nan
                    inverse[:, column]=inverse_empty_column
            ###Imputation on inverse
            inverse[1:] = self.imputer.transform(inverse[1:])
            ###NEW Round to integers for categorical columns
            inverse[1:, self.to_round] = np.rint(inverse[1:, self.to_round])
            print('ciao')


        #####NEW: remove values from the inverse neighborhood, following sample_removal parameter instructions
        if any(x != 0 for x in self.removal): #no need to do this if all are zeroes:
            for i, col in enumerate(inverse[1:].T):
                n = int(self.removal[i]*len(col))
                index_nan = np.random.choice(len(col), n, replace=False)
                col[index_nan] = np.nan
                data[1:].T[i][index_nan] = np.nan #remove on data as well.
        self.check = (data, inverse, self.removal) #CHECK
        #####ENDnew. This has to be the last thing we do to inverse.

        return data, inverse


class RecurrentTabularExplainer(LimeTabularExplainer):
    """
    An explainer for keras-style recurrent neural networks, where the
    input shape is (n_samples, n_timesteps, n_features). This class
    just extends the LimeTabularExplainer class and reshapes the training
    data and feature names such that they become something like

    (val1_t1, val1_t2, val1_t3, ..., val2_t1, ..., valn_tn)

    Each of the methods that take data reshape it appropriately,
    so you can pass in the training/testing data exactly as you
    would to the recurrent neural network.

    """

    def __init__(self, training_data, mode="classification",
                 training_labels=None, feature_names=None,
                 categorical_features=None, categorical_names=None,
                 kernel_width=None, kernel=None, verbose=False, class_names=None,
                 feature_selection='auto', discretize_continuous=True,
                 discretizer='quartile', random_state=None):
        """
        Args:
            training_data: numpy 3d array with shape
                (n_samples, n_timesteps, n_features)
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            kernel_width: kernel width for the exponential kernel.
            If None, defaults to sqrt(number of columns) * 0.75
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        # Reshape X
        n_samples, n_timesteps, n_features = training_data.shape
        training_data = np.transpose(training_data, axes=(0, 2, 1)).reshape(
                n_samples, n_timesteps * n_features)
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        if feature_names is None:
            feature_names = ['feature%d' % i for i in range(n_features)]

        # Update the feature names
        feature_names = ['{}_t-{}'.format(n, n_timesteps - (i + 1))
                         for n in feature_names for i in range(n_timesteps)]

        # Send off the the super class to do its magic.
        super(RecurrentTabularExplainer, self).__init__(
                training_data,
                mode=mode,
                training_labels=training_labels,
                feature_names=feature_names,
                categorical_features=categorical_features,
                categorical_names=categorical_names,
                kernel_width=kernel_width,
                kernel=kernel,
                verbose=verbose,
                class_names=class_names,
                feature_selection=feature_selection,
                discretize_continuous=discretize_continuous,
                discretizer=discretizer,
                random_state=random_state)

    def _make_predict_proba(self, func):
        """
        The predict_proba method will expect 3d arrays, but we are reshaping
        them to 2D so that LIME works correctly. This wraps the function
        you give in explain_instance to first reshape the data to have
        the shape the the keras-style network expects.
        """

        def predict_proba(X):
            n_samples = X.shape[0]
            new_shape = (n_samples, self.n_features, self.n_timesteps)
            X = np.transpose(X.reshape(new_shape), axes=(0, 2, 1))
            return func(X)

        return predict_proba

    def explain_instance(self, data_row, classifier_fn, labels=(1,),
                         top_labels=None, num_features=10, num_samples=5000,
                         distance_metric='euclidean', model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly perturbing features
        from the instance (see __data_inverse). We then learn locally weighted
        linear models on this neighborhood data to explain each of the classes
        in an interpretable way (see lime_base.py).

        Args:
            data_row: 2d numpy array, corresponding to a row
            classifier_fn: classifier prediction probability function, which
                takes a numpy array and outputs prediction probabilities. For
                ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults
                to Ridge regression in LimeBase. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        # Flatten input so that the normal explainer can handle it
        data_row = data_row.T.reshape(self.n_timesteps * self.n_features)

        # Wrap the classifier to reshape input
        classifier_fn = self._make_predict_proba(classifier_fn)
        return super(RecurrentTabularExplainer, self).explain_instance(
            data_row, classifier_fn,
            labels=labels,
            top_labels=top_labels,
            num_features=num_features,
            num_samples=num_samples,
            distance_metric=distance_metric,
            model_regressor=model_regressor)





