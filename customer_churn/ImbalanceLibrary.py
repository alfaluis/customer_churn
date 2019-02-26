from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids
import pandas as pd


class Imbalance:
    def __init__(self, return_indices):
        print('Imbalance object created')
        self.return_indices = return_indices

    def under_sampling(self, x, y, method='random'):
        # if the y data is dataframe, it must be flatted (num_sample, )
        if isinstance(y, pd.DataFrame):
            y_ = y.ravel()
        else:
            y_ = y

        # use tomeklinks under-sample
        if 'tomek' in str(method).lower():
            tl = TomekLinks(return_indices=True, ratio='majority')
            x_res, y_res, id_res = tl.fit_sample(x, y_)
        # Need to be implemented
        elif 'cluster' in str(method).lower():
            pass
        # the default option is use Random-Sample
        else:
            rus = RandomUnderSampler(return_indices=True)
            x_res, y_res, id_res = rus.fit_sample(x, y_)

        # return desired information
        if self.return_indices:
            return x_res, y_res, id_res
        return x_res, y_res
