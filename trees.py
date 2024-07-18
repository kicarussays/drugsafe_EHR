import pandas as pd
import numpy as np
import os
import pickle
import ray
import math
import matplotlib.pyplot as plt
# import pycaret
# from pycaret.classification import *
from tqdm import tqdm
from datetime import timedelta
from src.config import drug_col, lab_cols, demo_col, labdict, labnames, blood, vital, lab_demo_col, ccidict
from multiprocessing import Pool
import multiprocessing
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, f1_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


if os.path.exists('../usedata/cdiff/mldata.pkl'):
    with open('../usedata/cdiff/mldata.pkl', 'rb') as f:
        alldat = pickle.load(f)

else:
    with open('../usedata/cdiff/cdiff_timeline_vital.pkl', 'rb') as f:
        tl = pickle.load(f)
    with open('../usedata/cdiff/no_cdiff_timeline_vital.pkl', 'rb') as f:
        notl = pickle.load(f)


    with open('../usedata/cdiff/cdiff_demo_vital.pkl', 'rb') as f:
        cdiff_demo = pickle.load(f)
        cdiff_demo.set_index('person_id', inplace=True, drop=True)
        cdiff_demo = cdiff_demo.loc[[k for k in tl.keys()]]
        tl = {k: tl[k] for k in cdiff_demo.index}
    with open('../usedata/cdiff/no_cdiff_demo_vital.pkl', 'rb') as f:
        no_cdiff_demo = pickle.load(f)
        no_cdiff_demo.set_index('person_id', inplace=True, drop=True)
        no_cdiff_demo = no_cdiff_demo[no_cdiff_demo.index.isin(list(notl.keys()))]
        notl = {k: notl[k] for k in no_cdiff_demo.index}

    def demoset(cdiff_demo, cdiff):
        cdiff_demo.columns = [ccidict[col] if col in ccidict.keys() else col for col in cdiff_demo.columns]
        cdiff_demo['person_id'] = cdiff_demo.index
        cdiff_demo['cdiff'] = cdiff
        cdiff_demo = cdiff_demo[cdiff_demo.columns.difference(['cdiff_date', 'index_date', 'index-7', 'index+28', 'visit_source_value'], sort=False)]
        cdiff_demo.reset_index(drop=True, inplace=True)
        return cdiff_demo

    cdiff_demo, no_cdiff_demo = demoset(cdiff_demo, 1), demoset(no_cdiff_demo, 0)

    demo_cols = [
        'age', 'sex', 'tumor', 'MI', 'uncomplicated_diabetes',
        'kidney', 'complicated_diabetes', 'HF', 'metastatic', 'dimentia',
        'cerebrovascular', 'peripheral', 'pulmonary', 'liver', 'mild_liver',
        'ulcer', 'hemiplegia', 'tissue', 'HIV', 'anta', 'anti'
    ]

    dat = []
    for k, v in tqdm(tl.items()):
        _demo = cdiff_demo[cdiff_demo['person_id'] == k][demo_cols].reset_index(drop=True)
        _v = v.astype(float).interpolate(limit_direction='both', axis=1)
        dat.append(pd.concat([
            _demo, 
            pd.DataFrame(_v.iloc[:, -1][lab_cols]).T.reset_index(drop=True),
            pd.DataFrame(_v.iloc[:, 0][lab_cols]).T.reset_index(drop=True)], axis=1))
    dat = pd.concat(dat)
    dat.columns = demo_cols + lab_cols + [f'prev_{i}' for i in lab_cols]
    dat['cdiff'] = 1

    nodat = []
    for k, v in tqdm(notl.items()):
        _demo = no_cdiff_demo[no_cdiff_demo['person_id'] == k][demo_cols].reset_index(drop=True)
        nodat.append(pd.concat([
            _demo, 
            pd.DataFrame(v.iloc[:, -1][lab_cols]).T.reset_index(drop=True),
            pd.DataFrame(v.iloc[:, 0][lab_cols]).T.reset_index(drop=True)], axis=1))
    nodat = pd.concat(nodat)
    nodat.columns = demo_cols + lab_cols + [f'prev_{i}' for i in lab_cols]
    nodat['cdiff'] = 0


    alldat = pd.concat([dat, nodat])
    alldat.reset_index(drop=True, inplace=True)
    with open('../usedata/cdiff/mldata_impute.pkl', 'wb') as f:
        pickle.dump(alldat, f, pickle.HIGHEST_PROTOCOL)


with open('../usedata/cdiff/mldata_impute.pkl', 'rb') as f:
    alldat = pickle.load(f)
dat, nodat = alldat[alldat['cdiff'] == 1], alldat[alldat['cdiff'] == 0]
idx1, idx0 = list(range(len(dat))), list(range(len(nodat)))



param_grid = {
    'n_estimators': np.arange(20, 200, 20),
    'max_depth': list(np.arange(1, 11)) + [None],
    'max_features': list(np.arange(1, 11)) + ['auto'],
}

allresult_rfc = []
allresult_gbc = []
allresult_lgbc = []

for seed in range(60, 65):
    train_idx_1, val_idx_1 = train_test_split(idx1, test_size=0.4, random_state=seed)
    train_idx_0, val_idx_0 = train_test_split(idx0, test_size=0.4, random_state=seed)
    X_train = pd.concat([dat.iloc[train_idx_1], nodat.iloc[train_idx_0]])
    X_val = pd.concat([dat.iloc[val_idx_1], nodat.iloc[val_idx_0]])
    X_train, y_train = X_train.iloc[:, :-1], np.array(X_train.iloc[:, -1])
    X_val, y_val = X_val.iloc[:, :-1], np.array(X_val.iloc[:, -1])

    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(X_train)
    X_train = pd.DataFrame(imp.transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(imp.transform(X_val), columns=X_train.columns)

    clf = RandomForestClassifier()
    CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, verbose=0, n_jobs=64,
                          scoring='roc_auc')
    CV_clf.fit(X_train, y_train)
    result = pd.DataFrame(CV_clf.cv_results_)[[
        'params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score')
    allresult_rfc.append(result)

    clf = GradientBoostingClassifier()
    CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=2, verbose=0, n_jobs=64,
                          scoring='roc_auc')
    CV_clf.fit(X_train, y_train)
    result = pd.DataFrame(CV_clf.cv_results_)[[
        'params', 'mean_test_score', 'rank_test_score']].sort_values('rank_test_score')
    allresult_gbc.append(result)


_allresult_rfc = []
for i in allresult_rfc:
    _tmp = i
    _tmp['param_str'] = i['params'].apply(lambda x: f"{x['n_estimators']}_{x['max_depth']}_{x['max_features']}")
    _allresult_rfc.append(_tmp[['param_str', 'mean_test_score']])

first_rfc = _allresult_rfc[0]
for n, i in enumerate(_allresult_rfc):
    if n == 0: continue
    first_rfc = pd.merge(first_rfc, i, on='param_str', how='left')

first_rfc['score'] = first_rfc.iloc[:, 1:].mean(axis=1)
first_rfc.sort_values('score', ascending=False)


_allresult_gbc = []
for i in allresult_gbc:
    _tmp = i
    _tmp['param_str'] = i['params'].apply(lambda x: f"{x['n_estimators']}_{x['max_depth']}_{x['max_features']}")
    _allresult_gbc.append(_tmp[['param_str', 'mean_test_score']])

first_gbc = _allresult_gbc[0]
for n, i in enumerate(_allresult_gbc):
    if n == 0: continue
    first_gbc = pd.merge(first_gbc, i, on='param_str', how='left')

first_gbc['score'] = first_gbc.iloc[:, 1:].mean(axis=1)
first_gbc.sort_values('score', ascending=False)

first_gbc.to_csv('../results/gbc.csv', index=None)
first_rfc.to_csv('../results/rfc.csv', index=None)