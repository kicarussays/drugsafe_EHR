import pandas as pd
import numpy as np
import os
import pickle
import ray
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
# import pycaret
# from pycaret.classification import *
from tqdm import tqdm
from datetime import timedelta

from src.config import drug_col, demo_col, labdict, labnames, blood, vital, ccidict, lab_cols
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
pd.options.mode.chained_assignment = None
tqdm.pandas()
num_cpus=80

os.makedirs('../usedata/tmp', exist_ok=True)


def datediff(a, b, days=False):
    dif = pd.to_datetime(a) - pd.to_datetime(b)
    return dif.days if days else dif.dt.days    
    

def datebetween(a, b1, b2, inclusion='both'):
    if inclusion == 'both':
        if a >= b1 and a <= b2:
            return 1
        else:
            return 0
    elif inclusion == 'left':
        if a >= b1 and a < b2:
            return 1
        else:
            return 0
    elif inclusion == 'right':
        if a > b1 and a <= b2:
            return 1
        else:
            return 0
    elif inclusion == 'none':
        if a > b1 and a < b2:
            return 1
        else:
            return 0


dpath = '../data/cdiff'
dlist = os.listdir(dpath)
dlist = [
    'cdifficile.csv',
    'all_labs_IQR.csv',
    'antibiotics.csv',
    'cci.csv',
    'visit.csv'
]
data = {d.split('.')[0]: pd.read_csv(os.path.join(dpath, d)) for d in dlist}



lab_cols_dict = {
    'vital': ['SBP', 'DBP', 'HR', 'RR', 'BT'], 
    'cbc': ['WBC', 'Hb', 'PLT', 'Neutrophil', 'ANC'], 
    'liver': ['albumin', 'protein', 'Bilirubin', 'AST', 'ALP', 'ALT'], 
    'renal': ['BUN', 'Cr', 'Na', 'K', 'Cl', 'TCO2'], 
    'etc': ['CRP']
}

for group, lab_cols in lab_cols_dict.items():
    cdiff = data['cdifficile']
    labs = data['all_labs_IQR']
    anti = data['antibiotics']; anti = anti[anti['category'] != 'Metronidazole']
    visit = data['visit']
    cci = data['cci']

    with open('../usedata/cdiff/tmp/meanstd.pkl', 'rb') as f:
        meanstd = pickle.load(f)

    def cdifflabel(v):
        usev = str(v).lower()
        if 'pos' in usev or 'equi' in usev:
            return 'positive'
        elif 'neg' in usev:
            return 'negative'
        else:
            return 'exclude'

    cdiff = cdiff[['person_id', 'measurement_date', 'value_source_value']]
    cdiff['label'] = cdiff.apply(lambda x: cdifflabel(x['value_source_value']), axis=1)

    cdiff['date_rank'] = cdiff.groupby(['person_id'])['measurement_date'].rank(method='min')
    cdiff = cdiff[cdiff['date_rank'] == 1]
    cdiff['row'] = cdiff.groupby(['person_id']).cumcount()+1
    cdiff = cdiff[cdiff['row'] == 1][['person_id', 'measurement_date', 'value_source_value', 'label']]
    cdiff = cdiff[cdiff['label'] == 'positive'][['person_id', 'measurement_date']]
    cdiff['past28'] = pd.to_datetime(cdiff['measurement_date']) - timedelta(days=28)
    cdiff['past56'] = pd.to_datetime(cdiff['measurement_date']) - timedelta(days=56)

    ###### C.diff + Antibiotic ######
    # Merge Cdiff + Antibiotics
    print('Merge Cdiff + Antibiotics')
    ca = pd.merge(cdiff, anti, on=['person_id'], how='left')
    ca['measurement_date'] = pd.to_datetime(ca['measurement_date'])
    ca['drug_exposure_start_date'] = pd.to_datetime(ca['drug_exposure_start_date'])
    ca['drug_exposure_end_date'] = pd.to_datetime(ca['drug_exposure_end_date'])
    ca = ca[['person_id', 'measurement_date', 'past56', 'past28', 'age', 
            'gender_source_value', 'drug_exposure_start_date', 'drug_exposure_end_date', 
            'visit_occurrence_id', 'category', 'group']].sort_values(['person_id', 'category', 'drug_exposure_start_date'])
    ca['in28'] = ca.progress_apply(lambda x: datebetween(
        x['drug_exposure_start_date'], x['past28'], x['measurement_date'], inclusion='left'), axis=1)
    ca['days_supply'] = datediff(ca['drug_exposure_end_date'], ca['drug_exposure_start_date']) + 1

    # Select index date within 28 days from cdiff date 
    # Exclude if Cdiff occurred within 2 days from index date
    print('Select index date within 28 days from cdiff date')
    cain28 = ca[ca['in28'] == 1]
    cain28['drank'] = cain28.groupby(['person_id'])['drug_exposure_start_date'].rank(method='min')
    cain28 = cain28[cain28['drank'] == 1]
    cain28['row'] = cain28.groupby(['person_id']).cumcount()+1
    cain28 = cain28[cain28['row'] == 1][['person_id', 'measurement_date', 'drug_exposure_start_date']]
    cain28.columns = ['person_id', 'cdiff_date', 'index_date']
    cain28['index28'] = pd.to_datetime(cain28['index_date']) - timedelta(days=28)
    print(len(set(cain28['person_id'])))
    print('Exclude if Cdiff occurred within 3days from index date')
    cain28['diff'] = datediff(cain28['cdiff_date'], cain28['index_date'])
    cain28 = cain28[cain28['diff'] > 2]
    print(len(set(cain28['person_id'])))

    # Exclude drug interval less than 28 days + continuous case
    print('Exclude drug interval less than 28 days + continuous case')
    caout28 = ca[ca['person_id'].isin(cain28['person_id'])]
    caout28 = pd.merge(cain28, caout28, on=['person_id'], how='inner')
    caout28['endin28'] = caout28.progress_apply(lambda x: datebetween(
        x['drug_exposure_end_date'], x['index28'], x['index_date'], inclusion='left'), axis=1)
    exc_pid_28 = pd.unique(caout28[caout28['endin28'] == 1]['person_id'])
    exc_pid_cont = pd.unique(caout28[(caout28['drug_exposure_start_date'] < caout28['index_date']) & 
                                    (caout28['drug_exposure_end_date'] >= caout28['index_date'])]['person_id'])
    exc_pid = np.concatenate([exc_pid_28, exc_pid_cont])
    ca = ca[(ca['person_id'].isin(cain28['person_id'])) & (~ca['person_id'].isin(exc_pid))]
    ca['in28'] = ca.progress_apply(lambda x: datebetween(
        x['drug_exposure_start_date'], x['past28'], x['measurement_date'], inclusion='left'), axis=1)
    ca = ca[ca['in28'] == 1]
    print(len(set(ca['person_id'])))


    usevisit = visit[visit['visit_occurrence_id'].isin(list(ca['visit_occurrence_id']))][['visit_occurrence_id', 'visit_source_value']]
    ca = pd.merge(ca, usevisit, on=['visit_occurrence_id'], how='left')
    # ca = ca[ca['visit_source_value'] != '외래']
    plist = sorted(set(ca['person_id']))
    drugitems = np.sort(pd.unique(anti['category']))
    idx_date = ca.groupby(['person_id'])[['drug_exposure_start_date']].min()
    idx_date.reset_index(inplace=True)
    idx_date.columns = ['person_id', 'index_date']
    ca = pd.merge(ca, idx_date, on='person_id', how='inner')
    ca['index-7'] = ca['index_date'] - timedelta(days=7)
    ca['index+28'] = ca['index_date'] + timedelta(days=28)
    ca['cdiff_date'] = ca['measurement_date']

    ###### C.diff + Lab data ######
    # Filter lab between monitoring window
    uselab = labs[labs['person_id'].isin(plist)]
    cidx = ca[['person_id', 'cdiff_date', 'index_date', 'index-7', 'index+28']].drop_duplicates().index
    c_indexdate = ca.loc[cidx][['person_id', 'cdiff_date', 'index_date', 'index-7', 'index+28', 'visit_source_value']]
    cl = pd.merge(c_indexdate, uselab, on=['person_id'], how='left')
    cl['measurement_date'] = pd.to_datetime(cl['measurement_date'])
    cl = cl[(cl['measurement_date'].between(cl['index-7'], cl['index+28'])) & (cl['measurement_date'] <= cl['cdiff_date'])]

    mstd_labs = []
    for _lab in lab_cols:
        _tmp = cl[cl['labname'] == _lab]
        _mstd = meanstd.loc[_lab]['value_as_number']
        _tmp['value_as_number'] = (_tmp['value_as_number'] - _mstd['mean']) / _mstd['std']
        mstd_labs.append(_tmp)
    cl = pd.concat(mstd_labs)


    ### LABS

    pts = None
    for _lab in lab_cols:
        if pts != None:
            _tmp = cl[(cl['labname'] == _lab) & (cl['person_id'].isin(pts))]
        else:
            _tmp = cl[cl['labname'] == _lab]
        _tmp['7exist'] = _tmp.apply(lambda x: datebetween(x['measurement_date'], x['index-7'], x['index_date']), axis=1)
        _tmp = _tmp[_tmp['person_id'].isin(_tmp[_tmp['7exist'] == 1]['person_id'])]
        _tmp['28exist'] = _tmp.apply(lambda x: datebetween(x['measurement_date'], x['index_date'], x['index+28'], inclusion='none'), axis=1)
        _tmp = _tmp[_tmp['person_id'].isin(_tmp[_tmp['28exist'] == 1]['person_id'])]

        print(f"{_lab}: {len(set(_tmp['person_id']))}")
        pts = list(set(_tmp['person_id']))
    
    cl = cl.sort_values(['person_id', 'labname', 'measurement_date'])
    cl = cl[cl['person_id'].isin(pts)].sort_values(['person_id', 'labname', 'measurement_date'])

    ###### C.diff + CCI conditions ######
    usecci = cci[cci['person_id'].isin(pd.unique(cl['person_id']))]
    cc = pd.merge(c_indexdate, usecci, on=['person_id'], how='left')
    cc['condition_start_date'] = pd.to_datetime(cc['condition_start_date'])
    cc = cc[cc['condition_start_date'] < cc['index_date']]
    cc = cc[cc['category'] != 'Others']

    # Timeline and Case group demo construction
    @ray.remote
    def timeline_construction(chunk, ca, cl, cc):
        all_timeline = {}
        demo_df = pd.DataFrame(columns=['age', 'sex'] + list(pd.unique(cc['category'])), index=chunk)
        
        for _pid in chunk:
            useca = ca[ca['person_id'] == _pid]
            usecl = cl[cl['person_id'] == _pid]
            usecc = cc[cc['person_id'] == _pid]
            start_date, end_date = useca['index-7'].iloc[0], useca['index+28'].iloc[0]
            daterange = pd.date_range(start_date, end_date, inclusive='left')
            lab_timeline = pd.DataFrame(columns=daterange, index=lab_cols)
            drug_timeline = pd.DataFrame(0, columns=daterange, index=drugitems)

            for idx in useca.index:
                _useca = useca.loc[idx]
                _daterange = pd.date_range(_useca['drug_exposure_start_date'], _useca['drug_exposure_end_date'])
                drug_timeline.loc[_useca['category'], _daterange] = 1
                
            for idx in usecl.index:
                _usecl = usecl.loc[idx]
                lab_timeline.loc[_usecl['labname'], _usecl['measurement_date']] = _usecl['value_as_number']
                
            drug_timeline, lab_timeline = drug_timeline.loc[:, daterange], lab_timeline.loc[:, daterange]
            # lab_timeline = lab_timeline.astype(float).interpolate(axis=1, limit_direction='both').T
            # noise = np.random.normal(0, 0.01, lab_timeline.shape) 
            # lab_timeline = lab_timeline + noise
            timeline = pd.concat([lab_timeline, drug_timeline])
            timeline.loc['age', :] = useca.iloc[0]['age'] / 40 - 1
            timeline.loc['sex', :] = 0 if useca.iloc[0]['gender_source_value'] == 'F' else 1
            all_timeline[_pid] = timeline
            
            first_drug_date = useca.iloc[0]['drug_exposure_start_date']
            conds = pd.unique(cc[(cc['person_id'] == _pid) & (cc['condition_start_date'] <= first_drug_date)]['category'])
            demo_df.loc[_pid]['age'] = useca.iloc[0]['age'] / 40 - 1
            demo_df.loc[_pid]['sex'] = 0 if useca.iloc[0]['gender_source_value'] == 'F' else 1
            demo_df.loc[_pid, conds] = 1

        demo_df = demo_df.fillna(0)
        return [all_timeline, demo_df]

    ray.init(num_cpus=num_cpus)
    ray_ca, ray_cl, ray_cc = ray.put(ca), ray.put(cl), ray.put(cc)
    pid_chunk = np.array_split(pd.unique(cl['person_id']), num_cpus)
    chunk_list = [timeline_construction.remote(chunk, ray_ca, ray_cl, ray_cc) for chunk in pid_chunk]
    ray_chunk = ray.get(chunk_list)
    ray.shutdown()

    all_timeline = {}
    demo_df = []
    for tl in ray_chunk:
        for k, v in tl[0].items():
            all_timeline[k] = v
        demo_df.append(tl[1])
    demo_df = pd.concat(demo_df)

    labavgday = []
    labavgcnt = []
    for k in list(demo_df.index):
        labcount = (~all_timeline[k].loc[lab_cols].isna()).sum()
        labavgday.append(np.mean([0 if i == 0 else 1 for i in list(labcount)]))
        labavgcnt.append(np.mean(labcount))
    demo_df['labavgday'] = labavgday
    demo_df['labavgcnt'] = labavgcnt

    demo_df['person_id'] = demo_df.index
    demo_df = pd.merge(c_indexdate, demo_df, on=['person_id'], how='right')
    demo_df.index = demo_df['person_id']

    prevnull = []
    postnull = []
    for k in list(demo_df.index):
        idate = demo_df[demo_df['person_id'] == k]['index_date'].iloc[0]
        prevnull.append(all_timeline[k].loc[lab_cols, :idate].notnull().sum(axis=1).astype(bool).astype(int))
        postnull.append(all_timeline[k].loc[lab_cols, idate:].iloc[:, 1:].notnull().sum(axis=1).astype(bool).astype(int))

    prevnull = pd.concat(prevnull, axis=1).T
    prevnull.columns = [f'prev_{i}' for i in prevnull]
    postnull = pd.concat(postnull, axis=1).T
    postnull.columns = [f'post_{i}' for i in postnull]
    prevnull.reset_index(drop=True, inplace=True)
    postnull.reset_index(drop=True, inplace=True)
    demo_df.reset_index(drop=True, inplace=True)
    demo_df = pd.concat([demo_df, prevnull, postnull], axis=1)

    with open(f'../usedata/cdiff/cdiff_demo_{group}.pkl', 'wb') as f:
        pickle.dump(demo_df, f, pickle.HIGHEST_PROTOCOL)
    with open(f'../usedata/cdiff/cdiff_timeline_{group}.pkl', 'wb') as f:
        pickle.dump(all_timeline, f, pickle.HIGHEST_PROTOCOL)


    

    ################# NonCDI
    with open(f'../usedata/cdiff/cdiff_demo_{group}.pkl', 'rb') as f:
        demo_df = pickle.load(f)

    positive_ids = demo_df['person_id'] # Exclude for control
    noa = anti[~anti['person_id'].isin(pd.unique(positive_ids))]
    tmp = noa[['person_id', 'drug_exposure_start_date', 'drug_exposure_end_date']].groupby(
        ['person_id', 'drug_exposure_start_date']).progress_apply(lambda x: x['drug_exposure_end_date'].max())
        
    # index date processing
    tmp = pd.concat([
        pd.DataFrame(tmp).index.to_frame().reset_index(drop=True),
        pd.DataFrame(tmp, columns=['drug_exposure_end_date']).reset_index(drop=True)], 
        axis=1)

    # Exclude records with interval within 28 days (define index date)
    tmp['row'] = tmp.groupby('person_id').cumcount()+1
    tmp['row+1'] = tmp.groupby('person_id').cumcount()+2

    pid_only_1 = tmp[~tmp['person_id'].isin(tmp[tmp['row'] >= 2]['person_id'])]
    df_tmp = tmp[~tmp['person_id'].isin(pid_only_1['person_id'])]

    df1 = df_tmp[df_tmp.columns.difference(['row'], sort=False)]
    df2 = df_tmp[df_tmp.columns.difference(['row+1'], sort=False)]
    df1.columns = ['person_id', 'past_start_date', 'past_end_date', 'row']
    df = pd.merge(df2, df1, on=['person_id', 'row'], how='left')
    df['diff'] = datediff(df['drug_exposure_start_date'], df['past_end_date'])
    no_indexdate = df.drop(df[df['diff'] <= 28].index)
    no_indexdate = pd.concat([pid_only_1, no_indexdate])[[
        'person_id', 'drug_exposure_start_date']].sort_values([
        'person_id', 'drug_exposure_start_date'])
    no_indexdate.columns = ['person_id', 'index_date']

    # extract drug records for 28 days from index date
    noa = pd.merge(noa, no_indexdate, on=['person_id'], how='left')
    noa['index_date'] = pd.to_datetime(noa['index_date'])
    noa['index+28'] = noa['index_date'] + timedelta(days=28)
    noa['drug_exposure_start_date'] = pd.to_datetime(noa['drug_exposure_start_date'])

    def indexprocess(df):
        df['flag'] = df.apply(lambda x: datebetween(
            x['drug_exposure_start_date'], x['index_date'], x['index+28']
        ), axis=1)
        return df

    def parallel_indexprocess(df, func, n_cores=64):
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    noa_flag = parallel_indexprocess(noa, indexprocess)
    noa = noa_flag[noa_flag['flag'] == 1][noa_flag.columns.difference(['flag'], sort=False)]
    noa = noa[['person_id', 'age', 'gender_source_value', 'index_date', 'drug_exposure_start_date', 'drug_exposure_end_date', 'category']]
    noa['index+28'] = noa['index_date'] + timedelta(days=28)
    noa['index-7'] = noa['index_date'] - timedelta(days=7)


    # Lab name mapping
    uselab = labs[labs['person_id'].isin(pd.unique(noa['person_id']))]
    uselab['measurement_date'] = pd.to_datetime(uselab['measurement_date'])

    uselab = uselab[uselab.columns.difference(
        ['range_low', 'range_high', 'unit_source_value'], 
        sort=False)].dropna()
    
    # Lab name mapping
    uselab = labs[labs['person_id'].isin(pd.unique(noa['person_id']))]
    uselab['measurement_date'] = pd.to_datetime(uselab['measurement_date'])

    uselab = uselab[uselab.columns.difference(
        ['range_low', 'range_high', 'unit_source_value'], 
        sort=False)].dropna()

    no_indexdate = noa[['person_id', 'index_date', 'index+28', 'index-7']].drop_duplicates()
    no_indexdate = no_indexdate[no_indexdate['person_id'].isin(pd.unique(uselab['person_id']))].reset_index(drop=True)
    uselab = uselab[uselab['person_id'].isin(pd.unique(no_indexdate['person_id']))].reset_index(drop=True)

    lab_cols = [
        'SBP', 'DBP', 'HR', 'RR', 'BT', 'WBC', 'Hb', 'PLT', 'Neutrophil', 'ANC', 'albumin', 'protein',
        'Bilirubin', 'AST', 'ALP', 'ALT', 'BUN', 'Cr', 'Na', 'K', 'Cl', 'TCO2', 'CRP'
    ]
    uselab = uselab[uselab['labname'].isin(lab_cols)].reset_index(drop=True)
    nol = pd.merge(no_indexdate, uselab, on=['person_id'], how='left')
    nol['measurement_date'] = pd.to_datetime(nol['measurement_date'])
    nol = nol[(nol['measurement_date'].between(nol['index-7'], nol['index+28']))]

    mstd_labs = []
    for _lab in lab_cols:
        _tmp = nol[nol['labname'] == _lab]
        _mstd = meanstd.loc[_lab]['value_as_number']
        _tmp['value_as_number'] = (_tmp['value_as_number'] - _mstd['mean']) / _mstd['std']
        mstd_labs.append(_tmp)
    nol = pd.concat(mstd_labs)

    pts = None
    for _lab in lab_cols:
        if pts != None:
            _tmp = nol[(nol['labname'] == _lab) & (nol['person_id'].isin(pts))]
        else:
            _tmp = nol[nol['labname'] == _lab]
        _tmp['7exist'] = _tmp.apply(lambda x: datebetween(x['measurement_date'], x['index-7'], x['index_date']), axis=1)
        _tmp = _tmp[_tmp['person_id'].isin(_tmp[_tmp['7exist'] == 1]['person_id'])]
        _tmp['28exist'] = _tmp.apply(lambda x: datebetween(x['measurement_date'], x['index_date'], x['index+28'], inclusion='none'), axis=1)
        _tmp = _tmp[_tmp['person_id'].isin(_tmp[_tmp['28exist'] == 1]['person_id'])]

        print(f"{_lab}: {len(set(_tmp['person_id']))}")
        pts = list(set(_tmp['person_id']))
    

    nol = nol.sort_values(['person_id', 'labname', 'measurement_date'])
    nol = nol[nol['person_id'].isin(pts)].sort_values(['person_id', 'labname', 'measurement_date'])

    ###### C.diff + CCI conditions ######
    usecci = cci[cci['person_id'].isin(pd.unique(nol['person_id']))]
    cc = pd.merge(no_indexdate, usecci, on=['person_id'], how='left')
    cc['condition_start_date'] = pd.to_datetime(cc['condition_start_date'])
    cc = cc[cc['condition_start_date'] < cc['index_date']]
    cc = cc[cc['category'] != 'Others']

    import time
    s = time.time()
    noa.reset_index(drop=True, inplace=True)
    mstd_labs = []
    for _lab in pd.unique(nol['labname']):
        _tmp = nol[nol['labname'] == _lab]
        _mstd = meanstd.loc[_lab]['value_as_number']
        _tmp['value_as_number'] = (_tmp['value_as_number'] - _mstd['mean']) / _mstd['std']
        mstd_labs.append(_tmp)
    nol = pd.concat(mstd_labs)
    nol = nol[nol['labname'].isin(lab_cols)]

    no_indexdate = noa[['person_id', 'index_date', 'index+28', 'index-7']].drop_duplicates()
    no_indexdate['nrow'] = no_indexdate.groupby(['person_id']).cumcount()+1
    no_indexdate = no_indexdate[no_indexdate['nrow'] == 1][no_indexdate.columns.difference(['nrow'], sort=False)]
    no_indexdate = no_indexdate[no_indexdate['person_id'].isin(pd.unique(nol['person_id']))].reset_index(drop=True)
    nol = nol[nol['person_id'].isin(pd.unique(no_indexdate['person_id']))].reset_index(drop=True)
    noa = pd.merge(no_indexdate, 
                noa[noa.columns.difference(['index+28', 'index-7'], sort=False)], 
                on=['person_id', 'index_date'], 
                how='left')

    cci = pd.read_csv('/workspace/data/cdiff/cci_v1.csv')
    usecci = cci[cci['person_id'].isin(pd.unique(nol['person_id']))]
    cc = pd.merge(no_indexdate, usecci, on=['person_id'], how='left')
    cc['condition_start_date'] = pd.to_datetime(cc['condition_start_date'])
    cc = cc[cc['condition_start_date'] < cc['index_date']]
    cc = cc[cc['category'] != 'Others']
    drugitems = np.sort(pd.unique(noa['category']))
    print(f'Data Load Fin: {round((time.time() - s) / 60, 2)} mins')

    # Timeline and Case group demo construction
    @ray.remote
    def timeline_construction(chunk, ca, cl, cc):
        all_timeline = {}
        demo_df = pd.DataFrame(columns=['age', 'sex'] + list(pd.unique(cc['category'])), index=chunk)
        out_idx = []
        ca, cl, cc = ca[ca['person_id'].isin(chunk)], cl[cl['person_id'].isin(chunk)], cc[cc['person_id'].isin(chunk)]
        
        for _pid in chunk:
            useca = ca[ca['person_id'] == _pid]
            usecl = cl[cl['person_id'] == _pid]
            usecc = cc[cc['person_id'] == _pid]
            start_date, end_date = useca['index-7'].iloc[0], useca['index+28'].iloc[0]
            usecl = usecl[usecl['measurement_date'].between(start_date, end_date, inclusive='left')]
            if usecl.shape[0] == 0: out_idx.append(_pid); continue
            daterange = pd.date_range(start_date, end_date, inclusive='left')
            lab_timeline = pd.DataFrame(columns=daterange, index=lab_cols)
            drug_timeline = pd.DataFrame(0, columns=daterange, index=drugitems)

            for idx in useca.index:
                _useca = useca.loc[idx]
                _daterange = pd.date_range(_useca['drug_exposure_start_date'], _useca['drug_exposure_end_date'])
                drug_timeline.loc[_useca['category'], _daterange] = 1
            
            for idx in usecl.index:
                _usecl = usecl.loc[idx]
                lab_timeline.loc[_usecl['labname'], _usecl['measurement_date']] = _usecl['value_as_number']
                
            drug_timeline, lab_timeline = drug_timeline.loc[:, daterange], lab_timeline.loc[:, daterange]
            # lab_timeline = lab_timeline.astype(float).interpolate(axis=1, limit_direction='both').T
            # noise = np.random.normal(0, 0.01, lab_timeline.shape) 
            # lab_timeline = lab_timeline + noise
            timeline = pd.concat([lab_timeline, drug_timeline])
            timeline.loc['age', :] = useca.iloc[0]['age'] / 40 - 1
            timeline.loc['sex', :] = 0 if useca.iloc[0]['gender_source_value'] == 'F' else 1
            all_timeline[_pid] = timeline
            
            first_drug_date = useca.iloc[0]['drug_exposure_start_date']
            conds = pd.unique(cc[(cc['person_id'] == _pid) & (cc['condition_start_date'] <= first_drug_date)]['category'])
            demo_df.loc[_pid]['age'] = useca.iloc[0]['age'] / 40 - 1
            demo_df.loc[_pid]['sex'] = 0 if useca.iloc[0]['gender_source_value'] == 'F' else 1
            demo_df.loc[_pid, conds] = 1

        demo_df = demo_df.drop(out_idx)
        demo_df = demo_df.fillna(0)
        return [all_timeline, demo_df]

    num_cpus = 64
    ray.init(num_cpus=num_cpus)
    ray_ca, ray_cl, ray_cc = ray.put(noa), ray.put(nol), ray.put(cc)
    pid_chunk = np.array_split(pd.unique(nol['person_id']), num_cpus)
    chunk_list = [timeline_construction.remote(chunk, ray_ca, ray_cl, ray_cc) for chunk in pid_chunk]
    ray_chunk = ray.get(chunk_list)
    ray.shutdown()

    all_timeline = {}
    demo_df = []
    for tl in ray_chunk:
        for k, v in tl[0].items():
            all_timeline[k] = v
        demo_df.append(tl[1])
    demo_df = pd.concat(demo_df)

    labavgday = []
    labavgcnt = []
    for k in list(demo_df.index):
        labcount = (~all_timeline[k].loc[lab_cols].isna()).sum()
        labavgday.append(np.mean([0 if i == 0 else 1 for i in list(labcount)]))
        labavgcnt.append(np.mean(labcount))
    demo_df['labavgday'] = labavgday
    demo_df['labavgcnt'] = labavgcnt

    demo_df['person_id'] = demo_df.index
    demo_df = pd.merge(no_indexdate, demo_df, on=['person_id'], how='right')
    demo_df.index = demo_df['person_id']

    prevnull = []
    postnull = []
    for k in list(demo_df.index):
        idate = demo_df[demo_df['person_id'] == k]['index_date'].iloc[0]
        prevnull.append(all_timeline[k].loc[lab_cols, :idate].notnull().sum(axis=1).astype(bool).astype(int))
        postnull.append(all_timeline[k].loc[lab_cols, idate:].iloc[:, 1:].notnull().sum(axis=1).astype(bool).astype(int))

    prevnull = pd.concat(prevnull, axis=1).T
    prevnull.columns = [f'prev_{i}' for i in prevnull]
    postnull = pd.concat(postnull, axis=1).T
    postnull.columns = [f'post_{i}' for i in postnull]
    prevnull.reset_index(drop=True, inplace=True)
    postnull.reset_index(drop=True, inplace=True)
    demo_df.reset_index(drop=True, inplace=True)
    demo_df = pd.concat([demo_df, prevnull, postnull], axis=1)

    lastidx = demo_df[[col for col in demo_df.columns if 'prev' in col or 'post' in col]].sum(axis=1)
    demo_df = demo_df.loc[lastidx[lastidx == 46].index]
    lastpid = list(demo_df['person_id'])
    all_timeline = {k: v for k, v in all_timeline.items() if k in lastpid}

    tmp = pd.merge(cdiff, demo_df[demo_df['person_id'].isin(cdiff['person_id'])], on='person_id', how='right')
    tmp['diff'] = (pd.to_datetime(tmp['measurement_date']) - pd.to_datetime(tmp['index_date'])).dt.days
    demo_df = demo_df[~demo_df['person_id'].isin(tmp[tmp['diff'] <= 84]['person_id'])]
    lastpid = list(demo_df['person_id'])
    all_timeline = {k: v for k, v in tqdm(all_timeline.items()) if k in lastpid}

    with open(f'../usedata/cdiff/no_cdiff_demo_{group}.pkl', 'wb') as f:
        pickle.dump(demo_df, f, pickle.HIGHEST_PROTOCOL)
    with open(f'../usedata/cdiff/no_cdiff_timeline_{group}.pkl', 'wb') as f:
        pickle.dump(all_timeline, f, pickle.HIGHEST_PROTOCOL)
    

    from src.utils import timeline_data
    with open(f'../usedata/cdiff_demo_{group}.pkl', 'rb') as f:
        cdiff_demo = pickle.load(f)
    with open(f'../usedata/cdiff_timeline_{group}.pkl', 'rb') as f:
        cdiff_timeline = pickle.load(f)
    with open(f'../usedata/no_cdiff_demo_{group}.pkl', 'rb') as f:
        no_cdiff_demo = pickle.load(f)
    with open(f'../usedata/no_cdiff_timeline_{group}.pkl', 'rb') as f:
        no_cdiff_timeline = pickle.load(f)

        
    # All Demographics for PS Matching
    all_demo = pd.concat([cdiff_demo[cdiff_demo['labavgday'] > 0.03], 
                        no_cdiff_demo[no_cdiff_demo['labavgday'] > 0.03]])
    all_demo.columns = [ccidict[col] if col in ccidict.keys() else col for col in all_demo.columns]
    all_demo['cdiff'] = all_demo['cdiff_date'].notnull().astype(int)
    all_demo = all_demo[all_demo.columns.difference(['cdiff_date', 'index_date', 'index-7', 'index+28', 'visit_source_value'], sort=False)]
    cdiff_demo = cdiff_demo.set_index('person_id')
    no_cdiff_demo = no_cdiff_demo.set_index('person_id')

    _tl = dict()
    pids = list(all_demo['person_id'])
    for person_id, timeline in tqdm(cdiff_timeline.items()): 
        if person_id in pids:
            _tl[person_id] = timeline_data(timeline)

    with open(f'../usedata/cdiff_timeline_final_{group}.pkl', 'wb') as f:
        pickle.dump(_tl, f, pickle.HIGHEST_PROTOCOL)


    _tl = dict()
    pids = list(all_demo['person_id'])
    for person_id, timeline in tqdm(no_cdiff_timeline.items()): 
        if person_id in pids:
            _tl[person_id] = timeline_data(timeline)

    with open(f'../usedata/no_cdiff_timeline_final_{group}.pkl', 'wb') as f:
        pickle.dump(_tl, f, pickle.HIGHEST_PROTOCOL)


    anta = pd.read_csv('../data/antacid.csv')
    anta['drug_exposure_start_date'] = pd.to_datetime(anta['drug_exposure_start_date'])
    anta['drug_exposure_end_date'] = pd.to_datetime(anta['drug_exposure_end_date'])
    anid = pd.unique(anta['person_id'])

    cnt = 0
    anta_list = []
    anti_list = []
    for k in tqdm(cdiff_demo['person_id']):
        _anta = anta[anta['person_id'] == k]
        _dat = cdiff_demo[cdiff_demo['person_id'] == k].iloc[0]
        _tl = cdiff_timeline[k]
        cdiff_date = _dat['cdiff_date']
        cond1 = datebetween(_anta['drug_exposure_start_date'].min(), _dat['index-7'], cdiff_date, 'left')
        cond2 = datebetween(_anta['drug_exposure_start_date'].max(), _dat['index-7'], cdiff_date, 'left')

        anta_list.append(cond1 or cond2)
        anti_list.append(_tl.loc[drug_col].sum(axis=1).astype(bool).sum())

    cdiff_demo['anta'] = anta_list
    cdiff_demo['anti'] = anti_list

    with open(f'../usedata/cdiff_demo_{group}.pkl', 'wb') as f:
        pickle.dump(cdiff_demo, f, pickle.HIGHEST_PROTOCOL)
    with open(f'../usedata/no_cdiff_demo_{group}.pkl', 'wb') as f:
        pickle.dump(no_cdiff_demo, f, pickle.HIGHEST_PROTOCOL)

