def restore_column_order(array, modified_indexes):
    import numpy as np
    n = array.shape[1]
    original_indexes = np.zeros(n, dtype=int)
    original_indexes[modified_indexes] = np.arange(len(modified_indexes))
    original_indexes[np.setdiff1d(np.arange(n), modified_indexes)] = np.arange(len(modified_indexes), n)
    return array[:, original_indexes]

def print_test_row(results, test_row, mapping):
    # Get the first item of the results dictionary (assuming there's only one)
    result = next(iter(results.items()))
    # Get the order of the columns from the result
    column_order = [i[0] for i in result[1]]
    # Initialize an empty list to store the string values of the test row
    string_values = []
    # Iterate over the column order
    for i in column_order:
        if i in mapping:
        # Get the string value of the current column
            string_value = mapping[i][test_row[i]]
        else:
            string_value = test_row[i]
        feature_name = test_row.index[i]
        feature_value = result[1][column_order.index(i)][1]
        string_values.append((feature_name, string_value, feature_value))
    # Return the test row with the string values
    return string_values


#%load_ext autoreload
#%autoreload 2
#import limeMV4.lime.lime_tabular
import pandas as pd
import numpy as np
import random
import os
import pickle as pkl
#data
from sklearn import datasets
from sklearn import model_selection
#from sklearn.metrics import f1_score, recall_score, make_scorer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
#import xgboost as xgb
#xgb.set_config(verbosity=0)
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import lime
import lime.lime_tabular
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
import csv

import wget
wget.download('https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py')


import numpy as np
import pandas as pd
from utils import *
import torch
import seaborn as sns

import torch
import numpy as np

from scipy import optimize

def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.
    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.
    q : float
        Quantile level (starting from lower values).
    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.
    Returns
    -------
        quantiles : torch.DoubleTensor
    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.
    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.
    quant : float, default = 0.5
        Quantile to return (default is median).
    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.
    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.
    Returns
    -------
        epsilon: float
    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        MAE : float
    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.
    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.
    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)
    Returns
    -------
        RMSE : float
    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())

def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return 

##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.
    p : float
        Proportion of missing values to generate for variables which will have missing values.
    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.
    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).
    """

    n, d = X.shape

    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1) ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)
    ############################################################################QUESTO IF l'HO MESSO IO, INTERCEPTS ERA NONETYPE##############################
    if intercepts is not None: 
        ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts) 
    else:
        ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs))

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


# Function produce_NA for generating missing values ------------------------------------------------------

def produce_NA(X, p_miss, mecha="MCAR", opt=None, p_obs=None, q=None):
    """
    Generate missing values for specifics missing-data mechanism and proportion of missing values. 
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.
    p_miss : float
        Proportion of missing values to generate for variables which will have missing values.
    mecha : str, 
            Indicates the missing-data mechanism to be used. "MCAR" by default, "MAR", "MNAR" or "MNARsmask"
    opt: str, 
         For mecha = "MNAR", it indicates how the missing-data mechanism is generated: using a logistic regression ("logistic"), quantile censorship ("quantile") or logistic regression for generating a self-masked MNAR mechanism ("selfmasked").
    p_obs : float
            If mecha = "MAR", or mecha = "MNAR" with opt = "logistic" or "quanti", proportion of variables with *no* missing values that will be used for the logistic masking model.
    q : float
        If mecha = "MNAR" and opt = "quanti", quantile level at which the cuts should occur.
    
    Returns
    ----------
    A dictionnary containing:
    'X_init': the initial data matrix.
    'X_incomp': the data with the generated missing values.
    'mask': a matrix indexing the generated missing values.s
    """
    
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = X.astype(np.float32)
        X = torch.from_numpy(X)
    
    if mecha == "MAR":
        mask = MAR_mask(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "logistic":
        mask = MNAR_mask_logistic(X, p_miss, p_obs).double()
    elif mecha == "MNAR" and opt == "quantile":
        mask = MNAR_mask_quantiles(X, p_miss, q, 1-p_obs).double()
    elif mecha == "MNAR" and opt == "selfmasked":
        mask = MNAR_self_mask_logistic(X, p_miss).double()
    else:
        mask = (torch.rand(X.shape) < p_miss).double()
    
    X_nas = X.clone()
    X_nas[mask.bool()] = np.nan
    
    return {'X_init': X.double(), 'X_incomp': X_nas.double(), 'mask': mask}

def load_and_clean(dfname='adult'):
    #data = pd.read_csv('D:/TesiDS/Datasets/'+dfname+'.csv')
    data = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/Lancioprove/Datasets/'+dfname+'.csv')
    if dfname=='adult':
        data = data.rename(columns=lambda x: x.strip())
        target_name = 'class'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
        df.drop(['fnlwgt'], axis=1, inplace=True)
        col_names = [f for f in col_names if f != 'fnlwgt']
        df[target_name] = data[target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
    elif dfname == 'iris':
        target_name = 'class'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
        df[target_name] = data[target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
    elif dfname == 'titanic':
        target_name = 'Survived'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
        df.drop(['PassengerId', 'Name', 'Cabin', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)
        col_names = [feature for feature in df.columns if feature!=target_name]
        df['Age'] = df['Age'].fillna(np.mean(df['Age']))
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df[target_name] = data[target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
    elif dfname == 'german_credit':
        target_name = 'default'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
        df[target_name] = data[target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
    elif dfname == 'diabetes':
        target_name = 'Outcome'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
        df[target_name] = data[target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
    elif dfname == 'fico':
        target_name = 'RiskPerformance'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
        df[target_name] = data[target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
        ####here
    elif dfname =='compas-scores-two-years':
        col_names = ['age', 'age_cat', 'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']
        #feature selection
        df = data[col_names]
        #feature engineering
        df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])
        df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
        df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
        df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
        df['length_of_stay'] = np.abs(df['length_of_stay'])
        df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
        df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)
        df['length_of_stay'] = df['length_of_stay'].astype(int)
        df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)
        #class engineering
        def get_class(x):
            if x < 7:
                return 'Medium-Low'
            else:
                return 'High'
        df['class'] = df['decile_score'].apply(get_class)
        #feature selection
        del df['c_jail_in']
        del df['c_jail_out']
        del df['decile_score']
        del df['score_text']

        target_name = 'class'
        col_names = [feature for feature in df.columns if feature!=target_name]
        print('df.shape', df.shape)
        return df, target_name, col_names, dfname
    
def what_to_encode(dfname='adult'):
    if dfname == 'adult':
        to_encode = [1,2,4, 5,6,7,8,12] ###tolto 1 dopo il primo perchè ho tolto fnlwgt in load_and_clean
    elif dfname == 'titanic':
        to_encode = [0, 1, 4]
    elif dfname == 'german_credit':
        to_encode = [0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19]
    elif dfname == 'compas-scores-two-years':
        to_encode = [1,2,3,6,7,8,9]
    else:
        #iris: all continuous
        to_encode = []
    return to_encode
    
    
def label_encoding(df, dfname='adult', target_name='class'):
    
    categorical_names = {}
    categorical_names_number = {}
    to_encode = what_to_encode(dfname=dfname)
    categorical_features = df.columns[to_encode] #pass names like this to be quick
    le = LabelEncoder()
    for i, feature in enumerate(categorical_features):
        le = LabelEncoder()
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        categorical_names[feature] = le.classes_
        categorical_names_number[to_encode[i]] = le.classes_

    class_encoding = {}
    le_class = LabelEncoder()
    le_class.fit(df[target_name])
    df[target_name] = le_class.transform(df[target_name])
    class_encoding[target_name] = le_class.classes_
    
    return to_encode, categorical_features, le, categorical_names, categorical_names_number, class_encoding, le_class

def standardize(X_train, to_encode=[]):
    to_standardize = [x for x in range(len(X_train.columns)) if x not in to_encode]
    scaler_standard = ColumnTransformer([('Standard Scaler', StandardScaler(), 
                                        to_standardize)], remainder='passthrough')
    scaler_standard.fit(X_train.values)
    scaled_X_train = scaler_standard.transform(X_train.values)
    scaled_X_train = restore_column_order(scaled_X_train, to_standardize)
    
    return to_standardize, scaler_standard, scaled_X_train

def select_mv_strategy_paper(X_train, y_train, mv_cols_perc=None, standard=None):
    configurations = []
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    num_features = X_train.shape[1]
    if mv_cols_perc is not None:
        n = num_features * mv_cols_perc
        n = math.ceil(n)
    top_n = importances.argsort()[-n:][::-1]

    top_n_features = [X_train.columns[i] for i in top_n]
    top_n_features_idx = [i for i in range(len(X_train.columns)) if X_train.columns[i] in top_n_features]

    to_append = [standard]*num_features
    configurations.append({'idx': top_n_features_idx, 'perc': to_append})
    
    return configurations
    
def insert_mv_in_array(array, colidx, percentages, seed=42):
    df = array.copy()
    for (col, p) in zip(colidx, percentages):
        colseed = int(np.random.random()*100)
        sample_index = np.random.RandomState(seed=colseed).choice(df.shape[0], size=int(df.shape[0]*p), replace=False)
        df[sample_index,col] = np.nan
    return df

def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
          columns = {0 : 'Missing Values', 1 : '% of Total Values'}
        )
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
                '% of Total Values', ascending=False
            ).round(1)
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values."
        )
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

def remove_nans_onehot(array):
    pass

def train_hgb():
    pass

def hgb_cross_validation(encoded_X_train, y_train, to_encode, obj):
        '''parameters = {'base_score': [0.5], 'colsample_bylevel': [1], 'gamma': [0], 'max_delta_step':[0], 'n_estimators':[100], 
                      'num_boost_round' : [50], 'use_label_encoder' : [False], 'objective':[obj],
                      'colsample_bytree': [0.3], 'learning_rate': [0.1], 'max_depth': [2,3,4,5], 'alpha': [10]} 
                      #'num_class': [len(pd.value_counts(y_train))]}'''
        parameters = {'loss': ['log_loss'], 'learning_rate': [0.05, 0.1], 'max_iter': [100], 'max_leaf_nodes':[31, 51], 'max_depth':[None, 2], 
                      'min_samples_leaf' : [20, 50], 'l2_regularization' : [0.0], 'max_bins':[255],
                      'warm_start': [False], 'early_stopping': ['auto'], 'scoring': ['loss'], 'validation_fraction': [0.2],
                      'n_iter_no_change' : [10], 'tol': [1e-07]}
        hgb_base = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state=42)
        clf = GridSearchCV(hgb_base, parameters)
        clf.fit(encoded_X_train, y_train)
        '''results = pd.DataFrame(clf.cv_results_)
        results.sort_values(by='rank_test_score', inplace=True)
        results.loc[0, 'params']'''
        #MODEL
        hgb_model = clf.best_estimator_
        return hgb_model
    
def rf_cross_validation(encoded_X_train, y_train):
    clf = RandomForestClassifier(max_depth=5, random_state=42)
    return clf


def train_model(X_train, y_train, X_test, y_test, mymodel, verbose=False):
    #Train the Classifier and print performance
    model = mymodel
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    avg_score = sum(scores)/len(scores)
    if verbose == True:
        ypred = model.predict(X_test)
        print('Classification report:')
        print(classification_report(y_test, ypred))
    return model, avg_score


import math

def cycle_rows_paper(number_of_rows, test_dataset, explainer, predict_fn, colidx, percentages, path, start, class_encoding, target_name, trained_model, name='', lime='mv', regressor=None):
    for n, row in enumerate(test_dataset[:number_of_rows]):
        step = time.time()
        if lime == 'mv':
            exp = explainer.explain_instance(row, predict_fn, model_regressor=regressor, top_labels=1)
        elif lime == 'base':
            exp = explainer.explain_instance(row, predict_fn, model_regressor=None, top_labels=1, num_features=test_dataset.shape[1])
        for key in exp.local_exp:
            values = [round(x[1],5) for x in sorted(exp.local_exp[key], key=lambda x: x[0])]
            cfg_idx = [0]*len(values)
            for i, idx in enumerate(cfg_idx):
                if i in colidx:
                    cfg_idx[i] = round(percentages[colidx.index(i)], 2)
            mvs = [0]*len(values)
            #print('row:', row)
            for i, value in enumerate(row):
                if math.isnan(value):
                    print('FOUND A NAN')
                    mvs[i] = 1

            info = [cfg_idx, str(True)+str(True), n, mvs, name]

        to_append = values+info
        #########APPEND PREDICT PROBAS #################
        probas = []
        for j, __ in enumerate(class_encoding[target_name]):
            probas.append(round(trained_model.predict_proba(np.array(row).reshape(1, -1))[0][j], 4))
        #for j, pr in enumerate(explainer.local_probas):
        try:
            probas.append(round(list(exp.local_pred.values())[0][0], 4))
        except:
            probas.append(round(exp.local_pred[0], 4))
        try:
            probas.append(list(exp.local_exp.keys())[0])
            #
        except:
            probas.append(list(exp.local_pred.keys())[0])

        to_append = to_append + probas
        to_append.append(round(time.time()-step, 4))
        print(to_append[-2])

        with open(path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter="|")
            writer.writerow(to_append)
        print('Row', n, '-', row, 'explained in:', round(time.time()-step, 4))
        print('Seconds passed:', round(time.time()-start, 4), ' \n')


def cycle_rows(n, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test):
    for n, row in enumerate(test_dataset):
        step = time.time()
        if 'model_regressor' in comb.keys():
            exp = explainer.explain_instance(row, predict_fn, model_regressor=comb['model_regressor'], top_labels=1)
        else:
            exp = explainer.explain_instance(row, predict_fn, model_regressor=None, top_labels=1, num_features=test_dataset.shape[1])
        for key in exp.local_exp:
            # Use list comprehension to extract the second value of each tuple
            values = [round(x[1],5) for x in sorted(exp.local_exp[key], key=lambda x: x[0])]
            ###################################################### This can be done outside.
            cfg_idx = [0]*len(values)
            for i, idx in enumerate(cfg_idx):
                if i in config['idx']:
                    cfg_idx[i] = round(config['perc'][config['idx'].index(i)], 2)
            mvs = [0]*len(values)
            #print('row:', row)
            for i, value in enumerate(row):
                if math.isnan(value):
                    print('FOUND A NAN')
                    mvs[i] = 1
            #print('mvs', mvs)
                    
            info = [cfg_idx, str(comb['pos'][0])+str(comb['pos'][1]), n, mvs, comb['name']]
            ######################################################
        to_append = values+info
        #########APPEND PREDICT PROBAS #################
        probas = []
        for j, __ in enumerate(class_encoding[target_name]):
            probas.append(round(trained_model.predict_proba(np.array(row).reshape(1, -1))[0][j], 4))

        #for j, pr in enumerate(explainer.local_probas):
        try:
            probas.append(round(list(exp.local_pred.values())[0][0], 4))
        except:
            probas.append(round(exp.local_pred[0], 4))

        try:
            probas.append(list(exp.local_exp.keys())[0])
            #
        except:
            probas.append(list(exp.local_pred.keys())[0])


        to_append = to_append + probas

        to_append.append(round(time.time()-step, 4))

        print(to_append[-2])

        with open(path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter="|")
            writer.writerow(to_append)
        print('Row', n, '-', row, 'explained in:', round(time.time()-step, 4))
        print('Seconds passed:', round(time.time()-start, 4), ' \n')


#from fancyimpute import MICE

from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import lime_tabular as limeMV

def impute_missing_values(input_array, to_encode, imputation_method='mean'):
    # Create a copy of the input array to store the imputed values
    imputed_array = input_array.copy()
    if imputation_method == 'mean':
        imputers = []
        for i in range(input_array.shape[1]):
            column = input_array[:, i]
            if i in to_encode:
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            else:
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            column = imputer.fit_transform(column.reshape(-1, 1))
            imputed_array[:, i] = column.flatten()
            imputers.append(imputer)
        return imputed_array, imputers
    
    elif imputation_method == 'knn':
        imputers = []
        if len(to_encode)>0:
            imputer = KNNImputer(n_neighbors=1)
        else:
            imputer = KNNImputer(n_neighbors=5)
        imputed_array = imputer.fit_transform(input_array)
        imputers.append(imputer)    
        return imputed_array, imputers
    
    elif imputation_method == 'mice':
        imputers = []
        imp_num = IterativeImputer(#estimator=RandomForestRegressor(),
                               initial_strategy='mean',
                               max_iter=10, random_state=42)
        imp_cat = IterativeImputer(#estimator=RandomForestClassifier(), 
                               initial_strategy='most_frequent',
                               max_iter=10, random_state=42)
        df = pd.DataFrame(input_array)
        continuous = [x for x in list(df.columns) if x not in to_encode]
        df[continuous] = imp_num.fit_transform(df[continuous])
        if len(to_encode) > 0:
            df[to_encode] = imp_cat.fit_transform(df[to_encode])
        imputers.append(imp_num)
        imputers.append(imp_cat)
        return df.values, imputers
    else:
        raise ValueError('Invalid imputation method')


def save_rows_with_probas(X):
    pass
###################### START OF MAIN #############################
###################### START OF MAIN #############################
###################### START OF MAIN #############################
###################### START OF MAIN #############################
###################### START OF MAIN #############################
###################### START OF MAIN #############################
###################### START OF MAIN #############################

def prepare_datasets(namelist=['iris', 'titanic', 'german_credit', 'adult', 'diabetes', 'fico', 'compas-scores-two-years'], cross_validation=False, completely_random = True):
    start = time.time()
    dfnames_list = namelist
    for dfname in dfnames_list:
        df, target_name, col_names, dfname = load_and_clean(dfname = dfname)
        print('Dataset Loaded. Seconds passed:', round(time.time()-start, 4), '\n')
    
        to_encodeOLD, categorical_features, le, categorical_names, categorical_names_number, class_encoding, le_class = label_encoding(
                                                                                                            df, dfname=dfname,
                                                                                                            target_name=target_name)
        print('Categorical Features label encoded. Seconds passed:', round(time.time()-start, 4), '\n')
        to_encode = []

        X_train, X_test, y_train, y_test = model_selection.train_test_split(df[col_names], df[target_name], test_size=0.20, 
                                                                            random_state=42) 
        print('Train-Test Split Done. Seconds passed:', round(time.time()-start, 4), '\n')

        if len(pd.value_counts(y_train)) > 2:
            obj = 'multi:softmax'
        else:
            obj = 'binary:logistic'
        print(obj, 'chosen as objective function. Seconds passed:', round(time.time()-start, 4), '\n')    
        #our 30/50 test rows to explain! NEW
        number_of_rows = 50
        X_test_rows, y_test_rows = X_test.head(number_of_rows).copy(), y_test.head(number_of_rows).copy()
        print(number_of_rows, 'test rows selected. Seconds passed:', round(time.time()-start, 4), '\n')
        print('Here are the test rows selected:')
        print(X_test_rows, '\n')

        #standard scaler --> returns scaled_X_train which is not a df anymore, but an array. Needed for LIME.
        to_standardize, scaler_standard, scaled_X_train = standardize(X_train, to_encode=to_encode)
        print('Standard scaler fitted on X_train. Created the array scaled_X_train.  Seconds passed:', time.time()-start, ' \n')
        #Scalo sia test completo che test rows
        scaled_X_test = scaler_standard.transform(X_test.values)
        scaled_X_test = restore_column_order(scaled_X_test, to_standardize)
        scaled_X_test_rows = scaler_standard.transform(X_test_rows.values)
        scaled_X_test_rows = restore_column_order(scaled_X_test_rows, to_standardize)
        print('Test rows scaled with the Standard Scaler. Seconds passed:', round(time.time()-start, 4), '\n')

        ###SALVA train/test COMPLETI e PROCESSATI
        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_train_path = dir_path+'/Paper/Datasets/Base/'+dfname+'_base_train.csv'
        base_test_path = dir_path+'/Paper/Datasets/Base/'+dfname+'_base_test.csv'
        if os.path.exists(base_train_path):
            print(dfname+'_base_dataset already saved.')
        else:
            pd.DataFrame(scaled_X_train).to_csv(base_train_path, index=False, header=col_names)
            pd.DataFrame(scaled_X_test).to_csv(base_test_path, index=False, header=col_names)

        print('Creating CSV file for model performances. Seconds passed:', round(time.time()-start, 4), '\n')
        performance_path = dir_path+'/Paper/BlackBoxPerformance/'+dfname+'_performance.csv'
        output_names=["setting_name", "accuracy"]
        class_names = list(le_class.classes_)
        for cn in class_names:
            output_names.extend(["precision_"+str(cn), "recall_"+str(cn), "f1-score_"+str(cn), "support_"+str(cn)])
        if os.path.exists(performance_path):
            print("Path already existing. Just append new lines to an existing csv.")
        else:
            with open(performance_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output_names, delimiter="|")
                writer.writeheader()
        

        print('Creating CSV file for explanation results. Seconds passed:', round(time.time()-start, 4), '\n')
        #############CREATE CSV STRUCTURE
        output_names = []
        for name in X_train.columns:
            output_names.append(name)
        other_names = ['mv_perc', 'mv_loc', 'rowID', 'row_mv', 'method']
        for name in other_names:
            output_names.append(name)
        #########NEW PREDICT PROBAS##########
        for clname in class_encoding[target_name]:
            output_names.append('proba_'+str(clname)+'_class')
        #for clname in class_encoding[target_name]:
        output_names.append('predict_local')
        output_names.append('true_y')
        #output_names.append('true_y')
        output_names.append('time')
        print('These will be the columns of the output CSV:', output_names, '\n')

        results_path = dir_path+'/Paper/Results/'+dfname+'_results_PAPER.csv'
        if os.path.exists(results_path):
            print("Path already existing. Just append new lines to an existing csv.")
        else:
            with open(results_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output_names, delimiter="|")
                writer.writeheader()


        print('Entering Experimentation. Seconds passed:', round(time.time()-start, 4), '\n')
        ###Insert MCAR missing values and save datasets.
        if completely_random == True:
            col_percentages = [0.10, 0.20, 0.30, 0.40, 0.50]
            mv_percentages = [0.04, 0.08, 0.16, 0.32]
        else:
            col_percentages = [0.9, 0.8, 0.7]
            mv_percentages = [0.04, 0.08, 0.16, 0.32]
        for col_perc in col_percentages:
            if completely_random==True:
                print('Inserting MV in', col_perc, 'of columns. Seconds passed:', round(time.time()-start, 4), '\n')
                config = select_mv_strategy_paper(X_train, y_train, mv_cols_perc=col_perc, standard = mv_percentages[0])[0]
                colidx = [None]
                for mv_perc in mv_percentages:
                    print('Removing', col_perc, 'percent of each column. Seconds passed:', round(time.time()-start, 4), '\n')
                    if (colidx[0] is None) or (len(config['idx']) != len(colidx)):
                        colidx=config['idx']
                    percentages=[mv_perc]*len(config['idx'])
                    #print(colidx, percentages)
                    X_trainMV = insert_mv_in_array(scaled_X_train, colidx, percentages, seed=42)
                    X_testMV = insert_mv_in_array(scaled_X_test, colidx, percentages, seed=42) #used only for testing the model.
                    print('Missing values inserted. Seconds passed:', round(time.time()-start, 4), '\n')

                    #MV
                    ####################################################################################################################################################################
                    print('Saving dataset with MVs. Seconds passed:', round(time.time()-start, 4), '\n')
                    mv_train_path = dir_path+'/Paper/Datasets/Missing/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_missing_train_MCAR.csv'
                    mv_test_path = dir_path+'/Paper/Datasets/Missing/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_missing_test_MCAR.csv'
                    if os.path.exists(mv_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_base_dataset already saved.')
                    else:
                        pd.DataFrame(X_trainMV).to_csv(mv_train_path, index=False, header=col_names)
                        pd.DataFrame(X_testMV).to_csv(mv_test_path, index=False, header=col_names)
                    
                    #ALLENA MV MODEL!!!!! trained_hgbMV
                    print('Training model on incomplete data. Seconds passed:', round(time.time()-start, 4), '\n')
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelMV = hgb_cross_validation(X_trainMV, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelMV = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbMV, hgb_scoreMV = train_model(X_trainMV, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        X_testMV, y_test, hgb_modelMV, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreMV, 'Seconds passed:', round(time.time()-start, 4), ' \n')

                    ypred = trained_hgbMV.predict(X_testMV)
                    current_report = classification_report(y_test, ypred, output_dict = True)
                    append_performance = ["MV"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MCAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    #OUR METHOD (7)
                    print('Testing LimeMV. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'continuous', 
                                            tree_explainer= 'signs', 
                                            distance_type= 'continuous',
                                            custom_sample= 'knn', 
                                            sample_removal= 'auto',
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeMV_ContDist_KNN_TreeSigns'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor='tree')
                    
                    #OUR METHOD WITH MICE (7b)
                    print('Testing LimeMV. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'continuous', 
                                            tree_explainer= 'signs', 
                                            distance_type= 'continuous',
                                            custom_sample= 'mice', 
                                            sample_removal= 'auto',
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeMV_ContDist_MICE_TreeSigns'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor='tree')

                    #OUR METHOD WITH RUNIF IMP (7b)
                    

                    #Prova extra (8) - Il nostro metodo, ma con vicinato generato completamente (imputazione vicinato + distanze continue e surrogato albero)

                    print('Testing LimeMV with complete neighborhood. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'continuous', 
                                            tree_explainer= 'signs', 
                                            distance_type= 'continuous',
                                            custom_sample= 'knn', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeMVExtra_ContDist_KNN_TreeSigns'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor='tree')

                    #Prova extra (8b) - Il nostro metodo, ma con vicinato generato completamente (imputazione vicinato + distanze continue e surrogato albero) + MICE

                    print('Testing LimeMV with complete neighborhood. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'continuous', 
                                            tree_explainer= 'signs', 
                                            distance_type= 'continuous',
                                            custom_sample= 'mice', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeMVExtra_ContDist_MICE_TreeSigns'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor='tree')
                    
                    #Lime"BASE", ma implementato con MV, imputazione su vicinato con MEDIA (4)
                    print('Testing Lime Base Parameters, imputing in the neighborhood with MEAN. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'binary', 
                                            tree_explainer= False, 
                                            distance_type= 'binary',
                                            custom_sample= 'mean', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpNeighborhood_MEAN'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor=None)
                    

                    #Lime"BASE", ma implementato con MV, imputazione su vicinato con KNN (5)
                    print('Testing Lime Base Parameters, imputing in the neighborhood with KNN. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'binary', 
                                            tree_explainer= False, 
                                            distance_type= 'binary',
                                            custom_sample= 'knn', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpNeighborhood_KNN'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor=None)
                    

                    #Lime"BASE", ma implementato con MV, imputazione su vicinato con MICE (6)
                    print('Testing Lime Base Parameters, imputing in the neighborhood with MICE. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'binary', 
                                            tree_explainer= False, 
                                            distance_type= 'binary',
                                            custom_sample= 'mice', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpNeighborhood_MICE'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor=None)




                    ##train mean  #######################################################################################################################################################
                    print('Baseline. Impute directly on the dataset. With MEAN. Seconds passed:', round(time.time()-start, 4), '\n')
                    train_dataset = X_trainMV.copy()
                    test_dataset_mean = X_testMV.copy()
                    mean_train, imputers = impute_missing_values(train_dataset, to_encode, imputation_method='mean')
                    for i, col in enumerate(range(test_dataset_mean.shape[1])):
                        test_dataset_mean[:, col] = imputers[i].transform(test_dataset_mean[:, col].reshape(-1, 1)).reshape(-1)
                    mean_train_path = dir_path+'/Paper/Datasets/Mean/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MEAN_train_MCAR.csv'
                    mean_test_path = dir_path+'/Paper/Datasets/Mean/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MEAN_test_MCAR.csv'
                    #save train and test
                    if os.path.exists(mean_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_MEAN_dataset already saved.')
                    else:
                        pd.DataFrame(mean_train).to_csv(mean_train_path, index=False, header=col_names)
                        pd.DataFrame(test_dataset_mean).to_csv(mean_test_path, index=False, header=col_names)

                    #ALLENA MEAN MODEL!!!!! trained_hgbMEAN
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelMEAN = hgb_cross_validation(mean_train, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelMEAN = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbMEAN, hgb_scoreMEAN = train_model(mean_train, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        test_dataset_mean, y_test, hgb_modelMEAN, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreMEAN, 'Seconds passed:', round(time.time()-start, 4), ' \n')
                    ypredmean = trained_hgbMEAN.predict(X_testMV)
                    current_report = classification_report(y_test, ypredmean, output_dict = True)
                    append_performance = ["MEAN"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MCAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    explainer = lime.lime_tabular.LimeTabularExplainer(mean_train, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                    predict_fn = lambda x: trained_hgbMEAN.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpOutside_MEAN'
                    cycle_rows_paper(number_of_rows, test_dataset_mean, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='base', regressor=None)

                    #knn ##############################################################################################################################################################
                    print('Baseline. Impute directly on the dataset. With KNN. Seconds passed:', round(time.time()-start, 4), '\n')
                    train_dataset = X_trainMV.copy()
                    test_dataset_knn = X_testMV.copy()
                    knn_train, imputers = impute_missing_values(train_dataset, to_encode, imputation_method='knn')
                    test_dataset_knn = imputers[0].transform(test_dataset_knn)
                    knn_train_path = dir_path+'/Paper/Datasets/KNN/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_KNN_train_MCAR.csv'
                    knn_test_path = dir_path+'/Paper/Datasets/KNN/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_KNN_test_MCAR.csv'
                    #save train and test
                    if os.path.exists(knn_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_KNN_dataset already saved.')
                    else:
                        pd.DataFrame(knn_train).to_csv(knn_train_path, index=False, header=col_names)
                        pd.DataFrame(test_dataset_knn).to_csv(knn_test_path, index=False, header=col_names)

                    #ALLENA KNN MODEL!!!!! trained_hgbKNN
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelKNN = hgb_cross_validation(knn_train, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelKNN = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbKNN, hgb_scoreKNN = train_model(knn_train, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        test_dataset_knn, y_test, hgb_modelKNN, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreKNN, 'Seconds passed:', round(time.time()-start, 4), ' \n')
                    ypredKNN = trained_hgbKNN.predict(X_testMV)
                    current_report = classification_report(y_test, ypredKNN, output_dict = True)
                    append_performance = ["KNN"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MCAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    explainer = lime.lime_tabular.LimeTabularExplainer(knn_train, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                    predict_fn = lambda x: trained_hgbKNN.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpOutside_KNN'
                    cycle_rows_paper(number_of_rows, test_dataset_knn, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='base', regressor=None)

                    #mice ##############################################################################################################################################################
                    print('Baseline. Impute directly on the dataset. With MICE. Seconds passed:', round(time.time()-start, 4), '\n')
                    train_dataset = X_trainMV.copy()
                    test_dataset_mice = X_testMV.copy()
                    mice_train, imputers = impute_missing_values(train_dataset, to_encode, imputation_method='mice')
                    test_df_mice = pd.DataFrame(test_dataset_mice)
                    if len(to_encode)>0:
                        test_df_mice[to_encode] = imputers[1].transform(test_df_mice[to_encode])
                    test_df_mice[to_standardize] = imputers[0].transform(test_df_mice[to_standardize])
                    test_dataset_mice = test_df_mice.values
                    mice_train_path = dir_path+'/Paper/Datasets/MICE/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MICE_train_MCAR.csv'
                    mice_test_path = dir_path+'/Paper/Datasets/MICE/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MICE_test_MCAR.csv'
                    #save train and test
                    if os.path.exists(mice_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_MICE_dataset already saved.')
                    else:
                        pd.DataFrame(mice_train).to_csv(mice_train_path, index=False, header=col_names)
                        pd.DataFrame(test_dataset_mice).to_csv(mice_test_path, index=False, header=col_names)

                    #ALLENA MICE MODEL!!!!! trained_hgbKNN
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelMICE = hgb_cross_validation(mice_train, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelMICE = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbMICE, hgb_scoreMICE = train_model(mice_train, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        test_dataset_mice, y_test, hgb_modelMICE, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreMICE, 'Seconds passed:', round(time.time()-start, 4), ' \n')
                    ypredmice = trained_hgbMICE.predict(X_testMV)
                    current_report = classification_report(y_test, ypredmice, output_dict = True)
                    append_performance = ["MICE"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MCAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    explainer = lime.lime_tabular.LimeTabularExplainer(mice_train, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                    predict_fn = lambda x: trained_hgbMICE.predict_proba(x)
                    currname = 'MCAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpOutside_KNN'
                    cycle_rows_paper(number_of_rows, test_dataset_mice, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='base', regressor=None)

def prepare_datasetsMAR(namelist=['iris', 'titanic', 'german_credit', 'adult', 'diabetes', 'fico', 'compas-scores-two-years'], cross_validation=False, completely_random = False):
    start = time.time()
    dfnames_list = namelist
    for dfname in dfnames_list:
        np.random.seed(0)
        df, target_name, col_names, dfname = load_and_clean(dfname = dfname)
        print('Dataset Loaded. Seconds passed:', round(time.time()-start, 4), '\n')
    
        to_encodeOLD, categorical_features, le, categorical_names, categorical_names_number, class_encoding, le_class = label_encoding(
                                                                                                            df, dfname=dfname,
                                                                                                            target_name=target_name)
        print('Categorical Features label encoded. Seconds passed:', round(time.time()-start, 4), '\n')
        to_encode=[]

        X_train, X_test, y_train, y_test = model_selection.train_test_split(df[col_names], df[target_name], test_size=0.20, 
                                                                            random_state=42) 
        print('Train-Test Split Done. Seconds passed:', round(time.time()-start, 4), '\n')

        if len(pd.value_counts(y_train)) > 2:
            obj = 'multi:softmax'
        else:
            obj = 'binary:logistic'
        print(obj, 'chosen as objective function. Seconds passed:', round(time.time()-start, 4), '\n')    
        #our 30/50 test rows to explain! NEW
        number_of_rows = 50
        X_test_rows, y_test_rows = X_test.head(number_of_rows).copy(), y_test.head(number_of_rows).copy()
        print(number_of_rows, 'test rows selected. Seconds passed:', round(time.time()-start, 4), '\n')
        print('Here are the test rows selected:')
        print(X_test_rows, '\n')

        #standard scaler --> returns scaled_X_train which is not a df anymore, but an array. Needed for LIME.
        to_standardize, scaler_standard, scaled_X_train = standardize(X_train, to_encode=to_encode)
        print('Standard scaler fitted on X_train. Created the array scaled_X_train.  Seconds passed:', time.time()-start, ' \n')
        #Scalo sia test completo che test rows
        scaled_X_test = scaler_standard.transform(X_test.values)
        scaled_X_test = restore_column_order(scaled_X_test, to_standardize)
        scaled_X_test_rows = scaler_standard.transform(X_test_rows.values)
        scaled_X_test_rows = restore_column_order(scaled_X_test_rows, to_standardize)
        print('Test rows scaled with the Standard Scaler. Seconds passed:', round(time.time()-start, 4), '\n')

        ###SALVA train/test COMPLETI e PROCESSATI
        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_train_path = dir_path+'/Paper/Datasets/Base/'+dfname+'_base_train.csv'
        base_test_path = dir_path+'/Paper/Datasets/Base/'+dfname+'_base_test.csv'
        if os.path.exists(base_train_path):
            print(dfname+'_base_dataset already saved.')
        else:
            pd.DataFrame(scaled_X_train).to_csv(base_train_path, index=False, header=col_names)
            pd.DataFrame(scaled_X_test).to_csv(base_test_path, index=False, header=col_names)

        print('Creating CSV file for model performances. Seconds passed:', round(time.time()-start, 4), '\n')
        performance_path = dir_path+'/Paper/BlackBoxPerformance/'+dfname+'_performance.csv'
        output_names=["setting_name", "accuracy"]
        class_names = list(le_class.classes_)
        for cn in class_names:
            output_names.extend(["precision_"+str(cn), "recall_"+str(cn), "f1-score_"+str(cn), "support_"+str(cn)])
        if os.path.exists(performance_path):
            print("Path already existing. Just append new lines to an existing csv.")
        else:
            with open(performance_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output_names, delimiter="|")
                writer.writeheader()
        

        print('Creating CSV file for explanation results. Seconds passed:', round(time.time()-start, 4), '\n')
        #############CREATE CSV STRUCTURE
        output_names = []
        for name in X_train.columns:
            output_names.append(name)
        other_names = ['mv_perc', 'mv_loc', 'rowID', 'row_mv', 'method']
        for name in other_names:
            output_names.append(name)
        #########NEW PREDICT PROBAS##########
        for clname in class_encoding[target_name]:
            output_names.append('proba_'+str(clname)+'_class')
        #for clname in class_encoding[target_name]:
        output_names.append('predict_local')
        output_names.append('true_y')
        #output_names.append('true_y')
        output_names.append('time')
        print('These will be the columns of the output CSV:', output_names, '\n')

        results_path = dir_path+'/Paper/Results/'+dfname+'_results_PAPER.csv'
        if os.path.exists(results_path):
            print("Path already existing. Just append new lines to an existing csv.")
        else:
            with open(results_path, mode='w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=output_names, delimiter="|")
                writer.writeheader()


        print('Entering Experimentation. Seconds passed:', round(time.time()-start, 4), '\n')
        ###Insert MAR missing values and save datasets.
        if completely_random == True:
            col_percentages = [0.10, 0.20, 0.30, 0.40, 0.50]
            mv_percentages = [0.04, 0.08, 0.16, 0.32]
        else:
            col_percentages = [0.9, 0.8, 0.7]
            mv_percentages = [0.04, 0.08, 0.16, 0.32]
        for col_perc in col_percentages:
            if completely_random==False:
                print('Inserting MAR MV in', 1-col_perc, 'of columns. Seconds passed:', round(time.time()-start, 4), '\n')

                #config = select_mv_strategy_paper(X_train, y_train, mv_cols_perc=col_perc, standard = mv_percentages[0])[0]
                #colidx = [None]
                for mv_perc in mv_percentages:
                    print('Removing', col_perc, 'percent of each column. Seconds passed:', round(time.time()-start, 4), '\n')
                    #if (colidx[0] is None) or (len(config['idx']) != len(colidx)):
                    #    colidx=config['idx']
                    #percentages=[mv_perc]*len(config['idx'])
                    #print(colidx, percentages)
                    X_trainMV = produce_NA(scaled_X_train, p_miss=mv_perc, mecha='MAR', p_obs=col_perc)['X_incomp'].numpy()
                    X_testMV = produce_NA(scaled_X_test, p_miss=mv_perc, mecha='MAR', p_obs=col_perc)['X_incomp'].numpy()
                    #X_trainMV = insert_mv_in_array(scaled_X_train, colidx, percentages, seed=42)
                    #X_testMV = insert_mv_in_array(scaled_X_test, colidx, percentages, seed=42) #used only for testing the model.
                    print('Missing values inserted. Seconds passed:', round(time.time()-start, 4), '\n')

                    #MV
                    ####################################################################################################################################################################
                    print('Saving dataset with MVs. Seconds passed:', round(time.time()-start, 4), '\n')
                    mv_train_path = dir_path+'/Paper/Datasets/Missing/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_missing_train_MAR.csv'
                    mv_test_path = dir_path+'/Paper/Datasets/Missing/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_missing_test_MAR.csv'
                    if os.path.exists(mv_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_base_dataset already saved.')
                    else:
                        pd.DataFrame(X_trainMV).to_csv(mv_train_path, index=False, header=col_names)
                        pd.DataFrame(X_testMV).to_csv(mv_test_path, index=False, header=col_names)
                    
                    #ALLENA MV MODEL!!!!! trained_hgbMV
                    print('Training model on incomplete data. Seconds passed:', round(time.time()-start, 4), '\n')
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelMV = hgb_cross_validation(X_trainMV, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelMV = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbMV, hgb_scoreMV = train_model(X_trainMV, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        X_testMV, y_test, hgb_modelMV, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreMV, 'Seconds passed:', round(time.time()-start, 4), ' \n')

                    ypred = trained_hgbMV.predict(X_testMV)
                    current_report = classification_report(y_test, ypred, output_dict = True)
                    append_performance = ["MV"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)


                    #####SU MV devo provare: LimeMV completo + LimeMV con parametri di base x 3

                    #OUR METHOD (7)
                    print('Testing LimeMV. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'continuous', 
                                            tree_explainer= 'signs', 
                                            distance_type= 'continuous',
                                            custom_sample= 'knn', 
                                            sample_removal= 'auto',
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeMV_ContDist_KNN_TreeSigns'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor='tree')

                    #Prova extra (8) - Il nostro metodo, ma con vicinato generato completamente (imputazione vicinato + distanze continue e surrogato albero)

                    print('Testing LimeMV with complete neighborhood. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'continuous', 
                                            tree_explainer= 'signs', 
                                            distance_type= 'continuous',
                                            custom_sample= 'knn', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeMVExtra_ContDist_KNN_TreeSigns'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor='tree')
                    
                    #Lime"BASE", ma implementato con MV, imputazione su vicinato con MEDIA (4)
                    print('Testing Lime Base Parameters, imputing in the neighborhood with MEAN. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'binary', 
                                            tree_explainer= False, 
                                            distance_type= 'binary',
                                            custom_sample= 'mean', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpNeighborhood_MEAN'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor=None)
                    

                    #Lime"BASE", ma implementato con MV, imputazione su vicinato con KNN (5)
                    print('Testing Lime Base Parameters, imputing in the neighborhood with KNN. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'binary', 
                                            tree_explainer= False, 
                                            distance_type= 'binary',
                                            custom_sample= 'knn', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpNeighborhood_KNN'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor=None)
                    

                    #Lime"BASE", ma implementato con MV, imputazione su vicinato con MICE (6)
                    print('Testing Lime Base Parameters, imputing in the neighborhood with MICE. Seconds passed:', round(time.time()-start, 4), '\n')
                    explainer = limeMV.LimeTabularExplainerMV(X_trainMV, #select correct train.
                                            class_names=class_names,
                                                feature_names=col_names, categorical_features=to_encode,
                                                categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                            surrogate_data= 'binary', 
                                            tree_explainer= False, 
                                            distance_type= 'binary',
                                            custom_sample= 'mice', 
                                            sample_removal= None,
                                            to_round=to_encode)

                    predict_fn = lambda x: trained_hgbMV.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpNeighborhood_MICE'
                    cycle_rows_paper(number_of_rows, X_testMV, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='mv', regressor=None)




                    ##train mean  #######################################################################################################################################################
                    print('Baseline. Impute directly on the dataset. With MEAN. Seconds passed:', round(time.time()-start, 4), '\n')
                    train_dataset = X_trainMV.copy()
                    test_dataset_mean = X_testMV.copy()
                    mean_train, imputers = impute_missing_values(train_dataset, to_encode, imputation_method='mean')
                    for i, col in enumerate(range(test_dataset_mean.shape[1])):
                        test_dataset_mean[:, col] = imputers[i].transform(test_dataset_mean[:, col].reshape(-1, 1)).reshape(-1)
                    mean_train_path = dir_path+'/Paper/Datasets/Mean/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MEAN_train_MAR.csv'
                    mean_test_path = dir_path+'/Paper/Datasets/Mean/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MEAN_test_MAR.csv'
                    #save train and test
                    if os.path.exists(mean_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_MEAN_dataset already saved.')
                    else:
                        pd.DataFrame(mean_train).to_csv(mean_train_path, index=False, header=col_names)
                        pd.DataFrame(test_dataset_mean).to_csv(mean_test_path, index=False, header=col_names)

                    #ALLENA MEAN MODEL!!!!! trained_hgbMEAN
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelMEAN = hgb_cross_validation(mean_train, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelMEAN = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbMEAN, hgb_scoreMEAN = train_model(mean_train, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        test_dataset_mean, y_test, hgb_modelMEAN, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreMEAN, 'Seconds passed:', round(time.time()-start, 4), ' \n')
                    ypredmean = trained_hgbMEAN.predict(X_testMV)
                    current_report = classification_report(y_test, ypredmean, output_dict = True)
                    append_performance = ["MEAN"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    explainer = lime.lime_tabular.LimeTabularExplainer(mean_train, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                    predict_fn = lambda x: trained_hgbMEAN.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpOutside_MEAN'
                    cycle_rows_paper(number_of_rows, test_dataset_mean, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='base', regressor=None)

                    #knn ##############################################################################################################################################################
                    print('Baseline. Impute directly on the dataset. With KNN. Seconds passed:', round(time.time()-start, 4), '\n')
                    train_dataset = X_trainMV.copy()
                    test_dataset_knn = X_testMV.copy()
                    knn_train, imputers = impute_missing_values(train_dataset, to_encode, imputation_method='knn')
                    test_dataset_knn = imputers[0].transform(test_dataset_knn)
                    knn_train_path = dir_path+'/Paper/Datasets/KNN/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_KNN_train_MAR.csv'
                    knn_test_path = dir_path+'/Paper/Datasets/KNN/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_KNN_test_MAR.csv'
                    #save train and test
                    if os.path.exists(knn_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_KNN_dataset already saved.')
                    else:
                        pd.DataFrame(knn_train).to_csv(knn_train_path, index=False, header=col_names)
                        pd.DataFrame(test_dataset_knn).to_csv(knn_test_path, index=False, header=col_names)

                    #ALLENA KNN MODEL!!!!! trained_hgbKNN
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelKNN = hgb_cross_validation(knn_train, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelKNN = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbKNN, hgb_scoreKNN = train_model(knn_train, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        test_dataset_knn, y_test, hgb_modelKNN, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreKNN, 'Seconds passed:', round(time.time()-start, 4), ' \n')
                    ypredKNN = trained_hgbKNN.predict(X_testMV)
                    current_report = classification_report(y_test, ypredKNN, output_dict = True)
                    append_performance = ["KNN"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    explainer = lime.lime_tabular.LimeTabularExplainer(knn_train, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                    predict_fn = lambda x: trained_hgbKNN.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpOutside_KNN'
                    cycle_rows_paper(number_of_rows, test_dataset_knn, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='base', regressor=None)

                    #mice ##############################################################################################################################################################
                    print('Baseline. Impute directly on the dataset. With MICE. Seconds passed:', round(time.time()-start, 4), '\n')
                    train_dataset = X_trainMV.copy()
                    test_dataset_mice = X_testMV.copy()
                    mice_train, imputers = impute_missing_values(train_dataset, to_encode, imputation_method='mice')
                    test_df_mice = pd.DataFrame(test_dataset_mice)
                    if len(to_encode)>0:
                        test_df_mice[to_encode] = imputers[1].transform(test_df_mice[to_encode])
                    test_df_mice[to_standardize] = imputers[0].transform(test_df_mice[to_standardize])
                    test_dataset_mice = test_df_mice.values
                    mice_train_path = dir_path+'/Paper/Datasets/MICE/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MICE_train_MAR.csv'
                    mice_test_path = dir_path+'/Paper/Datasets/MICE/'+dfname+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_MICE_test_MAR.csv'
                    #save train and test
                    if os.path.exists(mice_train_path):
                        print(dfname+'_'+str(col_perc)+'_'+str(mv_perc)+'_MICE_dataset already saved.')
                    else:
                        pd.DataFrame(mice_train).to_csv(mice_train_path, index=False, header=col_names)
                        pd.DataFrame(test_dataset_mice).to_csv(mice_test_path, index=False, header=col_names)

                    #ALLENA MICE MODEL!!!!! trained_hgbKNN
                    if cross_validation:
                        print("Doing cross validaiton on complete data... please wait")
                        hgb_modelMICE = hgb_cross_validation(mice_train, y_train, to_encode=to_encode, obj=obj )
                    else:
                        hgb_modelMICE = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
                        print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
                    trained_hgbMICE, hgb_scoreMICE = train_model(mice_train, y_train, 
                                                        #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                                        test_dataset_mice, y_test, hgb_modelMICE, verbose=False)
                    print('Trained hgb, with CV score:', hgb_scoreMICE, 'Seconds passed:', round(time.time()-start, 4), ' \n')
                    ypredmice = trained_hgbMICE.predict(X_testMV)
                    current_report = classification_report(y_test, ypredmice, output_dict = True)
                    append_performance = ["MICE"+'_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_'+dfname+'_MAR']
                    append_performance.append(current_report['accuracy'])
                    for cl in range(len(class_encoding[target_name])):
                        for mtr in ['precision', 'recall', 'f1-score', 'support']:
                            append_performance.append(current_report[str(cl)][mtr])
                    with open(performance_path, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file, delimiter="|")
                        writer.writerow(append_performance)

                    explainer = lime.lime_tabular.LimeTabularExplainer(mice_train, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                    predict_fn = lambda x: trained_hgbMICE.predict_proba(x)
                    currname = 'MAR_'+str(int(col_perc*100))+'_'+str(int(mv_perc*100))+'_LimeBASE_ImpOutside_KNN'
                    cycle_rows_paper(number_of_rows, test_dataset_mice, explainer, predict_fn, colidx, percentages, results_path, start, class_encoding, target_name, 
                                    trained_hgbMV, name=currname, lime='base')


if __name__== '__main__':
    prepare_datasets(namelist=['iris', 'titanic'], cross_validation=True)
    #prepare_datasetsMAR(namelist=['iris'], cross_validation=False)

    #main(dfname='compas-scores-two-years', cross_validation=True)
    #main(dfname='german_credit', cross_validation=True)
    #main(dfname='diabetes')
    #main(dfname='fico', cross_validation=True)

    #main(dfname='iris') MCAR FATTO (ANDRE)
    #main(dfname='titanic', cross_validation=False)
    #main(dfname='adult', cross_validation=True)
