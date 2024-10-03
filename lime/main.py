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

def main():
    import pandas as pd
    import numpy as np
    import random
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
    import xgboost as xgb
    import time
    import lime_tabular

    adult = pd.read_csv("D:/TesiDS/Datasets/adult.csv")
    adult = adult.rename(columns=lambda x: x.strip())
    target_name = 'class'
    col_names = [feature for feature in adult.columns if feature!=target_name]
    df = pd.DataFrame(data=adult[col_names], columns=col_names)
    df[target_name] = adult[target_name]
    print('df.shape', df.shape)

    categorical_names = {}
    categorical_names_number = {}
    to_encode = [1,3,5, 6,7,8,9,13]
    categorical_features = df.columns[to_encode] #pass names like this to be quick
    for i, feature in enumerate(categorical_features):
        le = LabelEncoder()
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
        categorical_names[feature] = le.classes_
        categorical_names_number[to_encode[i]] = le.classes_

    class_encoding = {}
    le = LabelEncoder()
    le.fit(df[target_name])
    df[target_name] = le.transform(df[target_name])
    class_encoding[target_name] = le.classes_
    #print('class names', class_encoding)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(df[col_names], df[target_name], test_size=0.20, 
                                                                        random_state=42)
    
    #SELECT TEST ROWS HERE, FROM X_test
    #set random seed
    #choose n test rows and output as new array.

    #first, standard scaler
    to_standardize = [x for x in range(len(X_train.columns)) if x not in to_encode]
    scaler_standard = ColumnTransformer([('Standard Scaler', StandardScaler(), 
                                        to_standardize)], remainder='passthrough')
    scaler_standard.fit(X_train.values)
    scaled_X_train = scaler_standard.transform(X_train.values)
    scaled_X_train = restore_column_order(scaled_X_train, to_standardize)

    #here you could put a min max scaler if needed, with the same structure.

    #CREATE MISSING VALUES HERE, according to the chosen strategy:
    trainMV, testMV = False, False
    if trainMV:
        #put missing values in scaled_X_train (rename as new array, scaled_X_train_MV)
        #MVs in 
        pass
    if testMV:
        #standard scale the array with the 50 test rows
        #put missing values in there as well. (scaled_50_test_MV)
        pass


    #then, one hot encoder
    encoder_onehot = ColumnTransformer([('One-Hot Encoder', OneHotEncoder(handle_unknown='ignore'), 
                                        to_encode)], remainder='passthrough')
    encoder_onehot.fit(X_train.values)
    encoded_X_train = encoder_onehot.transform(scaled_X_train)
    #encoded_X_train = restore_column_order(encoded_X_train, to_encode)


    if len(pd.value_counts(y_train)) > 2:
        obj = 'multi:softmax'
    else:
        obj = 'binary:logistic'

    xgb_model = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
    gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10, n_estimators=100, 
    objective=obj, use_label_encoder=False)

    trained_model = xgb_model
    trained_model.fit(encoded_X_train, y_train)
    ypred = trained_model.predict(encoder_onehot.transform(scaler_standard.transform(X_test))) ##encode
    #print('Classification report:')
    #print(classification_report(y_test, ypred))

    predict_fn = lambda x: trained_model.predict_proba(encoder_onehot.transform(x)) #Qui è ok non scalare due volte?

    np.random.seed(1)

    class_names = [x.strip() for x in list(class_encoding['class'])]


    ###BUILD AN EXPLAINER WITH CERTAIN PARAMETERS
    explainer = lime_tabular.LimeTabularExplainerMV(scaled_X_train, 
                                    class_names=class_names,
                                      feature_names=col_names, categorical_features=to_encode,
                                      categorical_names=categorical_names_number, kernel_width=3, verbose=False,
                                    surrogate_data='continuous', #use 0-1 data to train regressor (like base LIME)
                                    tree_explainer=True, #don't use treeexplainer (since we still use the base regressor)
                                    distance_type='binary', #use 0-1 data to learn sample weights (like base LIME)
                                    custom_sample='continuous', #no effect as long as we don't have NaNs in the test row
                                    sample_removal=None) #do not insert missing values in the generated neighborhood


    np.random.seed(1)
    i = 15
    formatted_data_row = restore_column_order(scaler_standard.transform(X_test.iloc[i].values.reshape(1,-1)), 
                                          to_standardize).reshape(-1)
    exp = explainer.explain_instance(formatted_data_row, predict_fn, num_features=len(col_names), model_regressor='tree')

    local_exp = exp.local_exp
    printable = {}
    for el in local_exp:
        exp_list = []
        for couple in local_exp[el]:
            exp_list.append((col_names[couple[0]], couple[1], formatted_data_row[couple[0]]))
        printable[class_names[el]] = exp_list
        
    print(printable)
    output = print_test_row(local_exp, X_test.iloc[i], categorical_names_number)
    row = "{name1:^20}|{name2:^20}|{name3:^20}".format
    #maybe we can add the predicted class at the top for maximum clarity.
    print(row(name1='Feature', name2='Value', name3='Importance'))
    print(row(name1='', name2='', name3=''))
    for tup in output:
        print(row(name1=tup[0], name2=tup[1], name3=tup[2]))


    #wait
    print('end')
    





if __name__== '__main__':
    main()