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


def load_and_clean(dfname='adult'):
    #data = pd.read_csv('D:/TesiDS/Datasets/'+dfname+'.csv')
    data = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/Lancioprove/Datasets/'+dfname+'.csv')
    if dfname=='adult':
        data = data.rename(columns=lambda x: x.strip())
        target_name = 'class'
        col_names = [feature for feature in data.columns if feature!=target_name]
        df = pd.DataFrame(data=data[col_names], columns=col_names)
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
        to_encode = [1,3,5, 6,7,8,9,13]
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
    
def select_mv_strategy(X_train, y_train, double=True, all=False, max=None):
    configurations = []
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    num_features = X_train.shape[1]
    if all:
        n = num_features 
    else:
        n = num_features // 2
    top_n = importances.argsort()[-n:][::-1]
    #######VERSION_1
    # Normalize the importances to 100%
    importances = importances/importances.sum()#*100
    top_n_features = [X_train.columns[i] for i in top_n]
    top_n_features_idx = [i for i in range(len(X_train.columns)) if X_train.columns[i] in top_n_features]
    top_n_importances = [importances[i] for i in top_n]
    if max is not None:
        for i, imp in enumerate(top_n_importances):
            if imp > max:
                top_n_importances[i] = 0.5
    #if double == True:
    to_append = [x*2 for x in top_n_importances]
    for i, x in enumerate(to_append):
        if x > 0.5:
            to_append[i] = 0.5
        elif x < 0.15:
            to_append[i] = 0.15
    configurations.append({'idx': top_n_features_idx, 'perc': to_append})

    '''if double==True:
        configurations.append({'idx': top_n_features_idx, 'perc': [x*2 for x in top_n_importances]})'''
    
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

def combination_name(c):
    string = ''
    string = string + str(int(c['pos'][0]))+str(int(c['pos'][1]))+'_'
    if c['model_regressor'] is None:
        regressor = 'linear_'
    else:
        regressor = 'tree'
        if c['tree_explainer'] == False:
            regressor = regressor + 'XX_'
        elif c['tree_explainer'] == True:
            regressor = regressor + 'EX_'
        elif c['tree_explainer'] == 'sign' or c['tree_explainer'] == 'signs' :
            regressor = regressor + 'SN_'
    string = string + regressor
    if c['custom_sample'] is None:
        sample = 'none_'
    elif c['custom_sample'] == 'mean':
        sample = 'mean_'
    elif c['custom_sample'] == 'knn':
        sample = 'knnX_'
    elif c['custom_sample'] == 'normal':
        sample = 'norm_'
    elif c['custom_sample'] == 'uniform':
        sample = 'unif_'
    string = string+sample
    string = string+c['surrogate_data'][0:3]+'/'+c['distance_type'][0:3]+'_'
    if c['sample_removal'] == None:
        remov = 'none_'
    elif c['sample_removal'] == 'auto':
        remov = 'auto_'
    string = string+remov
    
    return string

def combination_name_base(c):
    string = '00_base_'
    if c['imputation_method'] == None:
        imp = 'full_'
    elif c['imputation_method'] == 'mean':
        imp = 'mean_'
    elif c['imputation_method'] == 'knn':
        imp = 'knnx_'
    elif c['imputation_method'] == 'extra':
        imp = 'extr_'
    string = string+imp+'imp_'
    if c['imputation_when'] is not None:
        string = string +c['imputation_when']
    else:
        string = string +'BASELINE'
    return string

def parameter_grid():
    grid = {(False, False): [], #BASELINES
           (True, False): [],
           (False, True): [],
           (True, True): []}
    
    pos = (False, False)
    #COMPARISON baselines
    '''new_parameters = {'model_regressor': None,
                      'tree_explainer': None,
                      'distance_type': 'binary',
                      'surrogate_data': 'binary',
                      'custom_sample': 'mean',
                      'sample_removal': None,
                      'name': '00_linear_COMPARISON_bin/bin',
                      'pos': (False, False)}
    
    grid[pos].append(new_parameters)'''

    new_parameters = {'model_regressor': None,
                      'tree_explainer': None,
                      'distance_type': 'continuous',
                      'surrogate_data': 'continuous',
                      'custom_sample': 'mean',
                      'sample_removal': None,
                      'name': '00_linear_COMPARISON_con/con',
                      'pos': (False, False)}
    
    grid[pos].append(new_parameters)
    
    new_parameters = {'model_regressor': None,
                      'tree_explainer': None,
                      'distance_type': 'continuous',
                      'surrogate_data': 'binary',
                      'custom_sample': 'mean',
                      'sample_removal': None,
                      'name': '00_linear_COMPARISON_con/bin',
                      'pos': (False, False)}
    
    grid[pos].append(new_parameters)
    
    new_parameters = {'model_regressor': 'tree',
                      'tree_explainer': True,
                      'distance_type': 'binary',
                      'surrogate_data': 'binary',
                      'custom_sample': 'mean',
                      'sample_removal': None,
                      'name': '00_treeEX_COMPARISON_bin/bin',
                      'pos': (False, False)}
    
    grid[pos].append(new_parameters)
    
    new_parameters = {'model_regressor': 'tree',
                      'tree_explainer': 'signs',
                      'distance_type': 'binary',
                      'surrogate_data': 'binary',
                      'custom_sample': 'mean',
                      'sample_removal': None,
                      'name': '00_treeSN_COMPARISON_bin/bin',
                      'pos': (False, False)}
    
    grid[pos].append(new_parameters)
    
    return grid
import math

def cycle_rows(n, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test):
    for n, row in enumerate(test_dataset):
        step = time.time()
        if 'model_regressor' in comb.keys():
            exp = explainer.explain_instance(row, predict_fn, model_regressor=comb['model_regressor'], top_labels=1)
        else:
            exp = explainer.explain_instance(row, predict_fn, model_regressor=None, top_labels=1)
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
    
    elif imputation_method == 'extra':
        imputers = []
        imp_num = IterativeImputer(#estimator=RandomForestRegressor(),
                               initial_strategy='mean',
                               max_iter=10, random_state=0)
        imp_cat = IterativeImputer(#estimator=RandomForestClassifier(), 
                               initial_strategy='most_frequent',
                               max_iter=10, random_state=0)
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


def main(dfname='iris', cross_validation=True): 
    start = time.time()   
    df, target_name, col_names, dfname = load_and_clean(dfname = dfname)
    print('Dataset Loaded. Seconds passed:', round(time.time()-start, 4), '\n')
    
    to_encode, categorical_features, le, categorical_names, categorical_names_number, class_encoding, le_class = label_encoding(
                                                                                                        df, dfname=dfname,
                                                                                                        target_name=target_name)
    print('Categorical Features label encoded. Seconds passed:', round(time.time()-start, 4), '\n')

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
    
    #choose how to insert the missing values. I'd do this directly on the numpy array versions.
    if dfname == 'iris':
        configurations = select_mv_strategy(X_train, y_train, double=False)
    else:
        configurations = select_mv_strategy(X_train, y_train, double=True)
    print(len(configurations), 'strategies for inserting MVs defined. Seconds passed:', round(time.time()-start, 4), ' \n')
    for cg in configurations:
        print(cg)
    print()

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

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path = dir_path+'/Lancioprove/Outputs/'+dfname+'_results_FINAL.csv'
    if os.path.exists(path):
        print("Path already existing. Just append new lines to an existing csv.")
    else:
        with open(path, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=output_names, delimiter="|")
            writer.writeheader()

    
    print('we are now going to enter in the testing phase. \n')
    for config in reversed(configurations):
        #ENTRA IN UN FOR CHE DURERA FINCHE NON CAMBI CONFIGURAZIONE!!
        colidx=config['idx']
        percentages=config['perc']
        print('In this configuration we removal values from columns:')
        print(colidx)
        print('According to distribution:')
        print([round(x*100, 2) for x in percentages], '\n')


        #CREATE MISSING VALUES HERE, according to the chosen strategy:
        X_trainMV = insert_mv_in_array(scaled_X_train, colidx, percentages, seed=42)
        X_testMV = insert_mv_in_array(scaled_X_test, colidx, percentages, seed=42) #used only for testing the model.
        #X_test_rowsMV = insert_mv_in_array(scaled_X_test_rows, colidx, percentages, seed=42) #my rows with MVs)
        X_test_rowsMV = X_testMV[:number_of_rows].copy()

        print('Values removed train and test. Created X_trainMV, X_testMV and X_test_rowsMV. Seconds passed:', round(time.time()-start, 4), ' \n')
        
        #ONEHOT-ENCODER: done only on the train set (useful only for training the model)
        #on complete train set
        #if len(to_encode) > 0:
        '''encoder_onehot = ColumnTransformer([('One-Hot Encoder', OneHotEncoder(handle_unknown='ignore'), 
                                        to_encode)], remainder='passthrough')
        encoder_onehot.fit(np.concatenate((scaled_X_train, scaled_X_test)))
        encoded_X_train = encoder_onehot.transform(scaled_X_train)
        print('One-Hot encoder fitted and applied for categorical features on the complete dataset. Seconds passed:', round(time.time()-start, 4), ' \n')'''
        '''else:
            encoded_X_train = scaled_X_train
            print('All features are continuous. No need to do one-hot encoding. Seconds passed:', round(time.time()-start, 4), ' \n')'''
        encoded_X_train = scaled_X_train
        print('Skipped One Hot Encoding on train')
        #on MV train set
        #if len(to_encode) > 0:
        '''encoder_onehotMV = ColumnTransformer([('One-Hot Encoder', OneHotEncoder(handle_unknown='ignore'), 
                                        to_encode)], remainder='passthrough')
        encoder_onehotMV.fit(np.concatenate((X_trainMV, X_testMV)))
        encoded_X_trainMV = encoder_onehotMV.transform(X_trainMV)  #already scaled and np array
        print('One-Hot encoder fitted and applied for categorical features on the dataset with MVs. Seconds passed:', round(time.time()-start, 4), ' \n')'''
        '''else:
            encoded_X_trainMV = X_trainMV
            print('All features are continuous. No need to do one-hot encoding. Seconds passed:', round(time.time()-start, 4), ' \n')'''
        encoded_X_trainMV = X_trainMV
        print('Skipped One Hot encoding on X_trainMV')
        #encoded_X_train = restore_column_order(encoded_X_train, to_encode)
        #DA FARE - TOGLI TUTTE LE COLONNE CHE Rappresentano dei NAN dallo one-hot encoding 
        
        #Train MODELS

        if cross_validation:
            #forest_model = rf_cross_validation(encoded_X_train, y_train)
            print("Doing cross validaiton on complete data... please wait")
            hgb_model = hgb_cross_validation(encoded_X_train, y_train, to_encode=to_encode, obj=obj )
            print("Doing cross validaiton on missing values data... please wait")
            hgb_modelMV = hgb_cross_validation(encoded_X_trainMV, y_train, to_encode=to_encode, obj=obj )
            print('Cross validation performed for RF (complete data) and hgb (both complete and MV data). Seconds passed:', round(time.time()-start, 4), ' \n')
        else:
            #forest_model = RandomForestClassifier(max_depth=5, random_state=0)
            '''hgb_model = hgb.hgbClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
            gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10, n_estimators=100, 
            objective=obj, use_label_encoder=False)'''
            hgb_model = HistGradientBoostingClassifier(categorical_features=to_encode, verbose=0, random_state = 42)
            print('Skipped Cross Validation. Using default parameters. Seconds passed:', round(time.time()-start, 4), ' \n')
               
        trained_hgb, hgb_score = train_model(encoded_X_train, y_train, 
                                             #encoder_onehot.transform(scaled_X_test), y_test, hgb_model, verbose=False)
                                             scaled_X_test, y_test, hgb_model, verbose=False)
        print('Trained hgb on complete data, with CV score:', hgb_score, 'Seconds passed:', round(time.time()-start, 4), ' \n')
        
        trained_hgbMV, hgbMV_score = train_model(encoded_X_trainMV, y_train, 
                                     X_testMV, y_test, hgb_model, verbose=False)
        print('Trained hgb on MV data, with CV score:', hgbMV_score, ' Seconds passed:', round(time.time()-start, 4), ' \n')
        
        '''trained_forest, forest_score = train_model(encoded_X_train, y_train, 
                                        #encoder_onehot.transform(scaled_X_test), y_test, forest_model, verbose=False)
                                        scaled_X_test, y_test, forest_model, verbose=False)
        print('Trained Random Forest on complete data, with CV score:', forest_score, ' Seconds passed:', round(time.time()-start, 4), ' \n')'''
        
        print('Given the current MV insertion strategy selected, we start testing various possible scenarios. MVs can be found:')
        print('- Nowhere (GROUND-TRUTH), or we make sure there are none by imputing them.')
        print('- Only in the training data.')
        print('- Only in the test row to explain.')
        print('- In both. \n')
        print('Recall that we are inserting MVS in this way:')
        print(colidx)
        print([round(x*100, 2) for x in percentages], '\n')





        #CHOOSE MV PLACEMENT
        MVplacements = [(False, False), (False, True), (True, False), (True, True)]
        '''if dfname == 'adult':
            MVplacements = [(True, False), (True, True)]'''
        #MVplacements = [(False, True), (True, True)]
        #MVplacements = [(False, True)]

        ######BUILD PARAMETER GRID (DO IN A A FUNCTION PASSING THE MVPLACEMENTS)
                  ##########NOE: placement FALSE FALSE are the baselines. Code appropriately.
        param_grid = parameter_grid()
        print('Parameter Grid built! Seconds passed:', round(time.time()-start, 4), ' \n')
        for p in param_grid:
            print('In case', p, 'we have', len(param_grid[p]), 'parameter combinations.')
        print('Now we will enter in a for cycle to test each parameter combination on each test row!')

        for pl in MVplacements:
            print('Dealing with the case where MVs are in:')
            print('Train_set:', pl[0])
            print('Test_rows:', pl[1], '\n')

            mv_in_train = pl[0]
            mv_in_test = pl[1]
            
            if mv_in_train == True or mv_in_test==True:
                trained_model = trained_hgbMV #onlyone I can use
                #trained_model.fit(encoded_X_trainMV, y_train) This isn't necessary since the models are already trained!!!
                #enc = encoder_onehotMV
                #ypred = trained_model.predict(enc.transform(X_testMV))
                ypred = trained_model.predict(X_testMV)
                print('Since there are MVs in the train test, the only model I can use is hgbOOST. Trained. Seconds passed:', round(time.time()-start, 4), ' \n')
            else:
                '''#enc = encoder_onehot
                if forest_score > hgb_score:
                    trained_model = trained_forest
                    print('The better model is Random Forest.')
                else:'''
                trained_model = trained_hgb
                print('The better model is hgboost.')
                
                #trained_model.fit(encoded_X_train, y_train) This isn't necessary since the models are already trained!!!
                #ypred = trained_model.predict(enc.transform(scaled_X_test)) ###da ricontrollare
                ypred = trained_model.predict(scaled_X_test)
                print('Trained the best model on the complete dataset. Seconds passed:', round(time.time()-start, 4), ' \n')
            print('Classification report for our classification model:')
            print(classification_report(y_test, ypred), '\n')
            
            '''if len(to_encode) > 0:
                predict_fn = lambda x: trained_model.predict_proba(enc.transform(x))
            else:
                predict_fn = lambda x: trained_model.predict_proba(x)'''
            predict_fn = lambda x: trained_model.predict_proba(x)

            np.random.seed(1)
            class_names = [str(x).strip() for x in list(class_encoding[target_name])]
            print('The class names are:', class_names, '\n')
            
            #####CYCLE OVER THE PARAMETER GRID
            for comb in param_grid[pl]: 
            #####FOR EACH PARAM COMBINATION:
                #SETUP data structure with a column for each feature, then CONFIG (mv%), then PL (location), 
                #                      , then a column for each parameter, and a synthetic name (ID) for a combination 
                #                        of parameters.
                #Need to setup the correct choice of train_test daset
                #####BUILD AN EXPLAINER with the parameters passed from grid
                if pl != (False, False):
                    if pl == (True, False):
                        #select train with MV 
                        train_dataset = X_trainMV
                        test_dataset = scaled_X_test_rows.copy()
                        #select rows complete 
                        
                    elif pl == (False, True):
                        #select train complete 
                        train_dataset = scaled_X_train
                        test_dataset = X_test_rowsMV.copy()
                        #select rows with MV   
                    
                    elif pl == (True, True):
                        #select train with MV 
                        train_dataset = X_trainMV
                        test_dataset = X_test_rowsMV.copy()
                        #select rows with MV 
                        # 
                    import lime_tabular as limeMV
                    explainer = limeMV.LimeTabularExplainerMV(train_dataset, #select correct train.
                                        class_names=class_names,
                                            feature_names=col_names, categorical_features=to_encode,
                                            categorical_names=categorical_names_number, kernel_width=3, verbose=False,

                                        surrogate_data= comb['surrogate_data'], 
                                        tree_explainer= comb['tree_explainer'], 
                                        distance_type= comb['distance_type'],
                                        custom_sample= comb['custom_sample'], 
                                        sample_removal= comb['sample_removal']) #model_regressor=None/'tree'
                    print('Built a LimeMV explainer for combination', comb['name']+'. Seconds passed:', round(time.time()-start, 4), ' \n')
                        
                    ######################HERE CYCLE OVER ROWS OF THE TEST DATASET
                    '''for n, row in enumerate(test_dataset):
                        step = time.time()
                        exp = explainer.explain_instance(row, predict_fn, model_regressor=comb['model_regressor'])
                        for key in exp.local_exp:
                            # Use list comprehension to extract the second value of each tuple
                            values = [round(x[1],5) for x in sorted(exp.local_exp[key], key=lambda x: x[0])]
                            ###################################################### This can be done outside.
                            cfg_idx = [0]*len(values)
                            for i, idx in enumerate(cfg_idx):
                                if i in config['idx']:
                                    cfg_idx[i] = round(config['perc'][config['idx'].index(i)], 2)
                            info = [cfg_idx, str(pl[0])+str(pl[1]), n, comb['name']]
                            ######################################################
                        to_append = values+info
                        with open(path, mode='a', newline='') as csv_file:
                            writer = csv.writer(csv_file, delimiter="|")
                            writer.writerow(to_append)
                        print('Row', n, '-', row, 'explained in:', round(time.time()-step, 4))
                        print('Seconds passed:', round(time.time()-start, 4), ' \n')'''
                    print(test_dataset)
                    cycle_rows(number_of_rows, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test)
                    #####################END CYCLE ROWS
                        #print('Local exp:', [round(x, 5) for x in to_append[0:len(values)]], '\n Seconds passed:', round(time.time()-start, 4), ' \n')
                    #print('\n finetestcicloMV')

                    ##############################################FALSE,FALSE
                else:
                    print('Case FALSE - FALSE - Baselines \n')
                    print(comb['name'])
                    trained_model = trained_hgb

                    #Baseline0 --> ground truth
                    if 'BASELINE' in comb['name']:
                        print('true baseline \n')
                        train_dataset = scaled_X_train
                        test_dataset = scaled_X_test_rows.copy()

                        '''if len(to_encode) > 0:
                            predict_fn = lambda x: trained_model.predict_proba(enc.transform(x))
                        else:
                            predict_fn = lambda x: trained_model.predict_proba(x)'''
                        predict_fn = lambda x: trained_model.predict_proba(x)

                        explainer = lime.lime_tabular.LimeTabularExplainer(train_dataset, #select correct train.
                                                    class_names=class_names, 
                                                    feature_names = col_names, categorical_features=to_encode, 
                                                   categorical_names=categorical_names_number, kernel_width=3, verbose=False)
                        print('Built a Base LIME explainer. Seconds passed:', round(time.time()-start, 4), ' \n')

                        ######################HERE CYCLE OVER ROWS OF THE TEST DATASET
                        cycle_rows(number_of_rows, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test)

                    elif 'COMPARISON' in comb['name']:
                        print('comparisons \n')
                        train_dataset = scaled_X_train
                        test_dataset = scaled_X_test_rows.copy()
                        import lime_tabular as limeMV
                        explainer = limeMV.LimeTabularExplainerMV(train_dataset, #select correct train.
                                        class_names=class_names,
                                            feature_names=col_names, categorical_features=to_encode,
                                            categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                        surrogate_data= comb['surrogate_data'], 
                                        tree_explainer= comb['tree_explainer'], 
                                        distance_type= comb['distance_type'],
                                        custom_sample= comb['custom_sample'], 
                                        sample_removal= comb['sample_removal'])
                        print('Built a LIMEMV explainer on complete data for testing. Seconds passed:', round(time.time()-start, 4), ' \n')
                        cycle_rows(number_of_rows, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test)

                    else:
                        train_dataset = X_trainMV
                        test_dataset = X_test_rowsMV.copy()
                        predict_fn = lambda x: trained_model.predict_proba(x)

                        #definisco i vari imputer
                        if comb['imputation_when'] == 'before':
                            train_dataset, imputers = impute_missing_values(train_dataset, to_encode, imputation_method=comb['imputation_method'])
                            if comb['imputation_method'] == 'knn':
                                test_dataset = imputers[0].transform(test_dataset)
                            elif comb['imputation_method'] == 'extra':
                                test_df = pd.DataFrame(test_dataset)
                                if len(to_encode)>0:
                                    test_df[to_encode] = imputers[1].transform(test_df[to_encode])
                                test_df[to_standardize] = imputers[0].transform(test_df[to_standardize])
                                test_dataset = test_df.values
                            else:
                                #assume mean
                                for i, col in enumerate(range(test_dataset.shape[1])):
                                    test_dataset[:, col] = imputers[i].transform(test_dataset[:, col].reshape(-1, 1)).reshape(-1)
                                

                            ####RETRAIN MODEL and set as trained_model
                            if len(to_encode) > 0:
                                '''encoder_onehot_baseline = ColumnTransformer([('One-Hot Encoder', OneHotEncoder(handle_unknown='ignore'), 
                                            to_encode)], remainder='passthrough')
                                encoder_onehot_baseline.fit(np.concatenate((train_dataset, test_dataset)))
                                encoded_train_dataset = encoder_onehot_baseline.transform(train_dataset)'''
                                encoded_train_dataset = train_dataset
                                trained_model, __ = train_model(encoded_train_dataset, y_train, 
                                                #encoder_onehot_baseline.transform(test_dataset), y_test.head(number_of_rows), hgb_model, verbose=False)
                                                test_dataset, y_test.head(number_of_rows), hgb_model, verbose=False)
                                        
                                #predict_fn = lambda x: trained_model.predict_proba(enc.transform(x))
                                predict_fn = lambda x: trained_model.predict_proba(x)
                
                            else:
                                trained_model, __ = train_model(train_dataset, y_train, 
                                                test_dataset, y_test.head(number_of_rows), hgb_model, verbose=False)
                                predict_fn = lambda x: trained_model.predict_proba(x)

                            explainer = lime.lime_tabular.LimeTabularExplainer(train_dataset, #select correct train.
                                                    class_names=class_names, 
                                                    feature_names = col_names, categorical_features=to_encode, 
                                                   categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                            print('Built a Base LIME explainer. Seconds passed:', round(time.time()-start, 4), ' \n')


                            ######################CYCLE
                            cycle_rows(number_of_rows, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test)

                        else:
                            #USE hgbOOST su train with mv
                            trained_model = trained_hgbMV
                            train_dataset, imputers = impute_missing_values(train_dataset, to_encode, imputation_method=comb['imputation_method'])
                            if comb['imputation_method'] == 'knn':
                                test_dataset = imputers[0].transform(test_dataset)
                            elif comb['imputation_method'] == 'extra':
                                test_df = pd.DataFrame(test_dataset)
                                test_df[to_encode] = imputers[1].transform(test_df[to_encode])
                                test_df[to_standardize] = imputers[0].transform(test_df[to_encode])
                                test_dataset = test_df.values
                            else:
                                #assume mean
                                for i, col in enumerate(range(test_dataset.shape[1])):
                                    test_dataset[:, col] = imputers[i].transform(test_dataset[:, col].reshape(-1, 1)).reshape(-1)

                            explainer = lime.lime_tabular.LimeTabularExplainer(train_dataset, #select correct train.
                                                    class_names=class_names, 
                                                    feature_names = col_names, categorical_features=to_encode, 
                                                   categorical_names=categorical_names_number, kernel_width=3, verbose=False)

                            print('Built a Base LIME explainer. Seconds passed:', round(time.time()-start, 4), ' \n')


                            ######################CYCLE
                            cycle_rows(number_of_rows, test_dataset, explainer, predict_fn, comb, config, path, start, class_encoding, target_name, trained_model, y_test)
                    
                    #Baseline1 --> In this case you also need to RETRAIN the classifiers.
                    #select train with MV --> Impute mean/mode o KNN --> BestClassifier --> LimeBase
                    #select rows with MV --> Impute mean/mode o KNN --> BestClassifier --> LimeBase
                    
                    #Baseline1 -->
                    #select train with MV --> hgboostClassifier --> Impute mean/mode o KNN --> LimeBase
                    #select rows with MV --> hgboostClassifier --> Impute mean/mode o KNN --> LimeBase
                    
                        '''explainer = lime.lime_tabular.LimeTabularExplainer(train_dataset, #select correct train.
                                                        class_names=class_names, 
                                                        feature_names = col_names, categorical_features=to_encode, 
                                                    categorical_names=categorical_names_number, kernel_width=3, verbose=False)
                        print('Built a Lime BASE explainer for combination', comb['name']+'. Seconds passed:', round(time.time()-start, 4), ' \n')
                        #take trainMV and impute, before or after training the class.model.
                        #explainer = lime BASE explainer'''

            
                ##### CYCLE OVER THE GROUP OF 30 APPROPRIATE ROWS AND EXPLAIN THEM, SAVING THE RESULT IN A CSV APPROPRIATELY.             
                        '''for row in test_dataset:
                            exp = 0###
                            print(row)'''




if __name__== '__main__':
    #main(dfname='iris')
    main(dfname='titanic', cross_validation=True)
    main(dfname='diabetes')
    main(dfname='compas-scores-two-years', cross_validation=True)
    main(dfname='adult', cross_validation=True)
    main(dfname='german_credit', cross_validation=True)
    main(dfname='fico', cross_validation=True)
