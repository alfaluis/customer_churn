import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve


def eliminate_spaces(df):
    try:
        aux = np.float32(df.TotalCharges)
    except ValueError as err:
        print(err)
        aux = np.nan
    return aux


def transform_columns(df):
    df_aux = df.copy()
    lb_enc = LabelEncoder()
    lb_bin = LabelBinarizer()
    processed_cols = list()
    for col in df_aux.columns:
        values = df_aux.loc[:, col].unique()
        num_values = len(values)
        print(num_values)
        if num_values == 2:
            df_aux.loc[:, col + 'Bin'] = lb_enc.fit_transform(df_aux.loc[:, col])
            processed_cols.append(col)
        elif 'Yes' in values and 'No' in values and values.shape[0] == 3:
            df_aux.loc[:, col] = df_aux.apply(lambda x: 'No' if 'No ' in x[col] else x[col], axis=1)
            df_aux.loc[:, col + 'Bin'] = lb_enc.fit_transform(df_aux.loc[:, col])
            processed_cols.append(col)
        elif 2 < num_values < 10:
            cols_bin = lb_bin.fit_transform(df_aux.loc[:, col])
            df_cols_bin = pd.DataFrame(cols_bin, columns=lb_bin.classes_)
            df_aux = pd.concat([df_aux, df_cols_bin], axis=1)
            processed_cols.append(col)
    return df_aux, processed_cols


def get_statistics(test):
    TP = test.loc[(test.ChurnBin == 1) & (test.Ypred == 1), 'ChurnBin'].shape[0]
    FP = test.loc[(test.ChurnBin == 0) & (test.Ypred == 1), 'ChurnBin'].shape[0]
    TN = test.loc[(test.ChurnBin == 0) & (test.Ypred == 0), 'ChurnBin'].shape[0]
    FN = test.loc[(test.ChurnBin == 1) & (test.Ypred == 0), 'ChurnBin'].shape[0]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    recall = TPR
    precision = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    return TPR, FPR, F1


if __name__ == '__main__':
    root = os.getcwd()
    database = os.path.join(root, 'database')
    df = pd.read_csv(os.path.join(database, 'CustomerChurn.csv'))
    df, processed_cols = transform_columns(df)
    # get only categorical and float data
    full_data = df.drop(columns=processed_cols, inplace=False)
    data = full_data.drop(columns='customerID', inplace=False)
    data.loc[:, 'TotalCharges'] = data.apply(eliminate_spaces, axis=1)
    data = data.dropna(subset=['TotalCharges'], axis=0, inplace=False).reset_index(drop=True)
    # return column name of column with float value
    col_2_norm = [col for col in data.columns if len(data[col].unique()) > 2]
    norm_col = [col + 'Norm' for col in col_2_norm]
    scaler = MinMaxScaler()
    df_norm_cols = pd.DataFrame(scaler.fit_transform(X=data.loc[:, col_2_norm]), columns=norm_col)
    data = pd.concat([data, df_norm_cols], axis=1)
    data.drop(columns=col_2_norm, inplace=True)

    logit = LogisticRegression(C=1.0, class_weight='balanced', dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)

    train, test = train_test_split(data, test_size=.25)
    train_y = train.loc[:, [x for x in train.columns if 'Churn' in x]]
    train_x = train.loc[:, [x for x in train.columns if not 'Churn' in x]]
    test_y = test.loc[:, [x for x in test.columns if 'Churn' in x]]
    test_x = test.loc[:, [x for x in test.columns if not 'Churn' in x]]

    logit.fit(train_x, train_y)
    predictions = logit.predict(test_x)
    prob = logit.predict_proba(test_x)
    test_y.loc[:, 'Ypred'] = predictions
    test_y.loc[:, 'Yprob'] = prob[:, 1]
    conf_matrix = confusion_matrix(test_y.loc[:, 'ChurnBin'], test_y.loc[:, 'Ypred'])
    fpr, tpr, thresholds = roc_curve(test_y.loc[:, 'ChurnBin'], test_y.loc[:, 'Yprob'])
    TPR, FPR, F1 = get_statistics(test=test_y)
    print(TPR, FPR, F1)
    print(conf_matrix)
    plt.plot(fpr, tpr)
    plt.grid()
    plt.show()

