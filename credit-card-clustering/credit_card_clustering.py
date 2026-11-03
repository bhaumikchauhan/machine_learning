import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

ALL_COLUMNS = ['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES',
               'INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY',
               'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
               'CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX',
               'CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']

LOG_COLUMNS = ['MINIMUM_PAYMENTS', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'PURCHASES',
               'CASH_ADVANCE', 'CASH_ADVANCE_TRX', 'PAYMENTS', 'PURCHASES_TRX',
               'BALANCE', 'CREDIT_LIMIT']

def get_data_from_csv():
    return pd.read_csv("kaggle\\input\\ccdata\\CC GENERAL.csv")

def build_pipeline(version=1):
    if version == 2:
        return Pipeline([('preprocess', build_preprocess_version_2()), ('model', KMeans(n_clusters=6, random_state=42))])
    preprocess = ColumnTransformer([
        ('minimum_payments', 
         Pipeline([
                    ('fill', SimpleImputer(strategy='constant', fill_value=0)),
                    ('log', FunctionTransformer(np.log1p)),
                    ('scale', StandardScaler()),
                ]), 
        ['MINIMUM_PAYMENTS']),
        ('credit_limit', 
         Pipeline([
                    ('fill', SimpleImputer(strategy='median')),
                    ('log', FunctionTransformer(np.log1p)),
                    ('scale', StandardScaler()),
                ]), 
        ['CREDIT_LIMIT']),
        ('cash_advance_frequency', 
         Pipeline([
                    ('clip', FunctionTransformer(np.clip, kw_args={'a_min': 0, 'a_max': 1})),
                    ('scale', StandardScaler()),
                ]), 
        ['CASH_ADVANCE_FREQUENCY']),
        ('log_and_scale', 
         Pipeline([
                    ('log', FunctionTransformer(np.log1p)),
                    ('scale', StandardScaler()),
         ]), 
         [col for col in LOG_COLUMNS if col not in ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']]),
        ('scale', StandardScaler(), 
         [col for col in ALL_COLUMNS if 
          col not in ['MINIMUM_PAYMENTS', 'CREDIT_LIMIT', 'CASH_ADVANCE_FREQUENCY'] 
          and col not in LOG_COLUMNS])
    ])


    return Pipeline([('preprocess', preprocess), ('model', KMeans(n_clusters=6, random_state=42))])


# Written by Claude (on request). Version 2: MINIMUM_PAYMENTS (and the one CREDIT_LIMIT blank) filled from the
# 5 nearest customers on BALANCE, CREDIT_LIMIT and PAYMENTS. Everything else is the same as version 1.
ACCOUNT_SIZE_COLUMNS = ['MINIMUM_PAYMENTS', 'BALANCE', 'CREDIT_LIMIT', 'PAYMENTS']

def build_preprocess_version_2():
    return ColumnTransformer([
        # log, then scale, then impute: KNNImputer finds neighbours by distance, so the columns must already be
        # on one scale. log1p keeps NaN as NaN, and StandardScaler ignores NaN when it learns the mean and sd.
        ('account_size',
         Pipeline([
                    ('log', FunctionTransformer(np.log1p)),
                    ('scale', StandardScaler()),
                    ('fill', KNNImputer(n_neighbors=5)),
                ]),
        ACCOUNT_SIZE_COLUMNS),
        ('cash_advance_frequency',
         Pipeline([
                    ('clip', FunctionTransformer(np.clip, kw_args={'a_min': 0, 'a_max': 1})),
                    ('scale', StandardScaler()),
                ]),
        ['CASH_ADVANCE_FREQUENCY']),
        ('log_and_scale',
         Pipeline([
                    ('log', FunctionTransformer(np.log1p)),
                    ('scale', StandardScaler()),
         ]),
         [col for col in LOG_COLUMNS if col not in ACCOUNT_SIZE_COLUMNS]),
        ('scale', StandardScaler(),
         [col for col in ALL_COLUMNS if
          col not in ACCOUNT_SIZE_COLUMNS + ['CASH_ADVANCE_FREQUENCY']
          and col not in LOG_COLUMNS])
    ])


def pca_coordinates():
    from sklearn.decomposition import PCA
    X = get_data_from_csv()
    X = X.drop(columns=['CUST_ID'])
    pipeline = build_pipeline()
    pipeline.fit(X)
    # reuses the fitted preprocessing: the same 17 scaled columns KMeans saw
    X_transformed = pipeline.named_steps['preprocess'].transform(X)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_transformed)
    labels = pipeline.named_steps['model'].labels_
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return coordinates, labels


def main():
    X_pca = pca_coordinates()
    # X = get_data_from_csv()
    # #drop the 'CUST_ID' column since it is not a feature
    # X = X.drop(columns=['CUST_ID'])
    # pipeline = build_pipeline()
    # pipeline.fit(X)
    # print("shape of the data after preprocessing:", pipeline.named_steps['preprocess'].transform(X).shape)
    # labels = pipeline.named_steps['model'].labels_
    # print(pd.Series(labels).value_counts(normalize=True))




if __name__ == '__main__':
    main()