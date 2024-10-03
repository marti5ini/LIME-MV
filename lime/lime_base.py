"""
Contains abstract functionality for learning locally linear sparse model.
"""
import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path
import lightgbm as lgb #NEW
from sklearn.utils import check_random_state

import collections
import copy
from functools import partial
import json
import warnings

####Imports from treeexplainer
import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from pyDOE2 import lhs
from scipy.stats.distributions import norm
####Imports from treeexplainer


def _get_tree_paths(tree, node_id, dft, depth=0):
    if 'L' in node_id:
        return [[node_id]]
    left_child = dft[dft['node_index'] == node_id]['left_child'].values[0]
    right_child = dft[dft['node_index'] == node_id]['right_child'].values[0]
    left_paths = _get_tree_paths(tree, left_child, dft, depth=depth + 1)
    right_paths = _get_tree_paths(tree, right_child, dft, depth=depth + 1)
    for path in left_paths:
        path.append(node_id)
    for path in right_paths:
        path.append(node_id)
    paths = left_paths + right_paths
    return paths

def get_importance(X, clf, X_train, y_train, objective='reg'):
    n_features = X_train.shape[1]
    dft = clf.booster_.trees_to_dataframe()
    dftk = dft[['node_index', 'split_feature']].copy()
    dftk = dftk.set_index('node_index')
    node_index2feature = dftk.to_dict()['split_feature']

    leaves_train = list()
    leaves_train_id_list = clf.booster_.predict(X_train, pred_leaf=True)
    for v in leaves_train_id_list:
        leaves_train.append('0-L%d' % v)

    paths = _get_tree_paths(clf, '0-S0', dft, depth=0)
    for path in paths:
        path.reverse()

    leaf_to_path = dict()
    for path in paths:
        leaf_to_path[path[-1]] = path

    if objective == 'clf': 
        n_classes = len(np.unique(y_train))

    values_dict = dict()
    for i in range(len(X_train)):
        path_i = leaf_to_path[leaves_train[i]]
        for node_id in path_i:

            if objective == 'reg':
                if node_id not in values_dict:
                    values_dict[node_id] = list()
                values_dict[node_id].append(y_train[i])

            if objective == 'clf': 
                if node_id not in values_dict:
                    values_dict[node_id] = np.zeros(n_classes)
                values_dict[node_id][y_train[i]] += 1

    if objective == 'reg':
        for k, v in values_dict.items():
            #values_dict[k] = np.mean(v)
            values_dict[k] = (np.mean(v)-np.mean(y_train))/np.std(y_train)

    if objective == 'clf':
        for k, v in values_dict.items():
            values_dict[k] = v/np.sum(v)

    unique_leaves = np.unique(leaves_train)
    unique_contributions = dict()

    for row, leaf in enumerate(unique_leaves):
        for path in paths:
            if leaf == path[-1]:
                break

        contribs = dict()
        for i in range(len(path) - 1):
            contrib = values_dict[path[i+1]] - values_dict[path[i]]
            if node_index2feature[path[i]] not in contribs:
                if objective == 'reg':
                    contribs[node_index2feature[path[i]]] = 0.0
                if objective == 'clf':
                    contribs[node_index2feature[path[i]]] = np.zeros(n_classes)
            contribs[node_index2feature[path[i]]] += contrib
        unique_contributions[leaf] = contribs

    # fino qui puo' essere fatto una volta sola dopo aver costruito il modello
    # qui applicato al test

    leaves = list()
    leaves_id_list = clf.booster_.predict(X, pred_leaf=True)
    for v in leaves_id_list:
        leaves.append('0-L%d' % v)

    contributions = list()
    for row, leaf in enumerate(leaves):
        contributions.append(unique_contributions[leaf])

    contributions_np = list()
    for i in range(len(contributions)):
        if objective == 'reg':
            contribs_np = np.zeros(n_features)
        if objective == 'clf':
            contribs_np = np.zeros((n_features, n_classes))
        for j in range(n_features):
            feat_name = 'Column_%d' % j
            if feat_name in contributions[i]:
                contribs_np[j] = contributions[i][feat_name]
        contributions_np.append(contribs_np)
    contributions_np = np.array(contributions_np)

    return contributions_np[0] ##################


class LimeBase(object):
    """Class for learning a locally linear sparse model from perturbed data"""
    def __init__(self,
                 kernel_fn,
                 verbose=False,
                 random_state=None):
        """Init function

        Args:
            kernel_fn: function that transforms an array of distances into an
                        array of proximity values (floats).
            verbose: if true, print local prediction values from linear model.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.kernel_fn = kernel_fn
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

    @staticmethod
    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=self.random_state)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0.01, fit_intercept=True,
                        random_state=self.random_state)
            clf.fit(data, labels, sample_weight=weights)

            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)



    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None,
                                   tree_explainer = False):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label] #yss
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection) #set as None, so always using all features.
        if model_regressor is None:
            model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=self.random_state)

        easy_model = model_regressor
        tree_flag = False
        ###########
        if model_regressor == 'tree' or np.isnan(neighborhood_data[1:,:]).any():
            tree_flag = True
            easy_model = lgb.LGBMRegressor(num_leaves=31, max_depth=3, learning_rate=1.0, 
                        n_estimators=1,
                        #objective='binary', 
                        importance_type='gain', 
                        boosting_type='rf', 
                        bagging_freq=1, 
                        bagging_fraction=0.9999, 
                        force_col_wise=True,
                        verbosity=-1)
        ###########

        self.checkneighdata = neighborhood_data
        if (np.isnan(neighborhood_data[0,:]).any()) and (model_regressor!='tree'):
            easy_model.fit(neighborhood_data[1:, used_features], #to completely remove feature selection, act here.
                        labels_column[1:], sample_weight=weights[1:])

            prediction_score = easy_model.score(
            neighborhood_data[1:, used_features],
            labels_column[1:], sample_weight=weights[1:])
            local_pred = easy_model.predict(np.nan_to_num(neighborhood_data[0, used_features], nan=1).reshape(1, -1))
        else:
            easy_model.fit(neighborhood_data[:, used_features], #to completely remove feature selection, act here.
                        labels_column, sample_weight=weights)
            prediction_score = easy_model.score(
                neighborhood_data[:, used_features],
                labels_column, sample_weight=weights)

            local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        #local_probas = easy_model.predict_proba(neighborhood_data[0,used_features])
        self.local_proba = local_pred

        ###########
        if model_regressor == 'tree'or tree_flag == True:
            if self.verbose:
                print('Prediction_local', local_pred,)
                print('Right:', neighborhood_labels[0, label])

            if tree_explainer==True: #use professor's tree explainer for lightgbm
                fi = get_importance(neighborhood_data[:1], easy_model, neighborhood_data, labels_column, 'reg')
                return (0,
                    sorted(zip(used_features, fi), #l'output di tree explainer va al posto di featureimportance.
                        key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred)

            elif tree_explainer=='signs' or tree_explainer=='sign':
                fi = get_importance(neighborhood_data[:1], easy_model, neighborhood_data, labels_column, 'reg')
                imp = easy_model.feature_importances_
                for i, x in enumerate(imp):
                    if fi[i]<0:
                        imp[i] = x*-1
                return (0,
                    sorted(zip(used_features, imp), #lfeature importance con segno aggiustato.
                        key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred)


            else: #return simple feature importances
                return (0,
                    sorted(zip(used_features, easy_model.feature_importances_), #l'output di tree explainer va al posto di featureimportance.
                        key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred)
        ###########
        else:
            if self.verbose:
                print('Intercept', easy_model.intercept_)
                print('Prediction_local', local_pred,)
                print('Right:', neighborhood_labels[0, label])
            return (easy_model.intercept_,
                    sorted(zip(used_features, easy_model.coef_),
                        key=lambda x: np.abs(x[1]), reverse=True),
                    prediction_score, local_pred)
