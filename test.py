import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA


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


def plot_2d_space(x, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            x[y == l, 0],
            x[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


def train_and_test_model(x_train, y_train, x_test, y_test, c, weight=None):
    if not len(y_train.shape) == 1:
        y_train_flat = y_train.ChurnBin.ravel()
    else:
        y_train_flat = y_train

    y = y_test.copy()
    logit = LogisticRegression(C=c, class_weight=weight, dual=False, fit_intercept=True,
                               intercept_scaling=1, max_iter=1000, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)

    logit.fit(x_train, y_train_flat)
    predictions = logit.predict(x_test)
    prob = logit.predict_proba(x_test)
    y.loc[:, 'Ypred'] = predictions
    y.loc[:, 'Yprob'] = prob[:, 1]
    conf_matrix = confusion_matrix(y.loc[:, 'ChurnBin'], y.loc[:, 'Ypred'])
    fpr, tpr, thresholds = roc_curve(y.loc[:, 'ChurnBin'], y.loc[:, 'Yprob'])
    # TPR, FPR, F1 = get_statistics(test=test_y)
    return {'model': logit, 'predictions': y, 'cm': conf_matrix, 'fpr': fpr, 'tpr': tpr, 'thres': thresholds}


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

    train, test = train_test_split(data, test_size=.25)
    train_y = train.loc[:, [x for x in train.columns if 'Churn' in x]]
    train_x = train.loc[:, [x for x in train.columns if 'Churn' not in x]]
    test_y = test.loc[:, [x for x in test.columns if 'Churn' in x]]
    test_x = test.loc[:, [x for x in test.columns if 'Churn' not in x]]

    # implement a 3d visualization
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(train_x)
    # plot_2d_space(X, train_y.ChurnBin)

    # training with a classic method - Just Logistic regression
    model_1 = train_and_test_model(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y, c=1, weight=None)

    # Train with classic method but using class weighting
    model_2 = train_and_test_model(x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y, c=1, weight='balanced')

    # train with classic method but using balanced class
    rus = RandomUnderSampler(return_indices=True)
    X_rus, y_rus, id_rus = rus.fit_sample(train_x, train_y.ChurnBin.ravel())
    print('Shape before under-sampling: ', train_x.shape)
    print('Shape before class Churn == True == 1: ', train_y.loc[train_y.ChurnBin == 1, :].shape)
    print('Shape before class Churn == False == 0: ', train_y.loc[train_y.ChurnBin == 0, :].shape)
    print('Shape after under-sampling: ', X_rus.shape)
    print('Shape after class Churn == True == 1: ', sum(y_rus == 1))
    print('Shape after class Churn == False == 0: ', sum(y_rus == 0))
    model_3 = train_and_test_model(x_train=X_rus, y_train=y_rus, x_test=test_x, y_test=test_y, c=1, weight=None)

    ros = RandomOverSampler(return_indices=True)
    X_ros, y_ros, id_ros = ros.fit_sample(train_x, train_y.ChurnBin.ravel())
    print('Shape before under-sampling: ', train_x.shape)
    print('Shape before class Churn == True == 1: ', train_y.loc[train_y.ChurnBin == 1, :].shape)
    print('Shape before class Churn == False == 0: ', train_y.loc[train_y.ChurnBin == 0, :].shape)
    print('Shape after under-sampling: ', X_ros.shape)
    print('Shape after class Churn == True == 1: ', sum(y_ros == 1))
    print('Shape after class Churn == False == 0: ', sum(y_ros == 0))
    model_4 = train_and_test_model(x_train=X_ros, y_train=y_ros, x_test=test_x, y_test=test_y, c=1, weight=None)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(model_1['fpr'], model_1['tpr'], label='Simple')
    ax[0].plot(model_2['fpr'], model_2['tpr'], label='Weighted')
    ax[0].plot(model_3['fpr'], model_3['tpr'], label='UnderSample')
    ax[0].plot(model_4['fpr'], model_4['tpr'], label='OverSample')
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title('ROC')
    ax[1].plot(model_1['fpr'], model_1['thres'], label='Simple')
    ax[1].plot(model_2['fpr'], model_2['thres'], label='Weighted')
    ax[1].plot(model_3['fpr'], model_3['thres'], label='UnderSample')
    ax[1].plot(model_4['fpr'], model_4['thres'], label='OverSample')
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title('Threshold')
    plt.show()
