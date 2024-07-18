import os
import pickle
import ray
import math
import torch
import numpy as np
from numpy import argmax
import pandas as pd
# import pycaret
# from pycaret.classification import *
from tqdm import tqdm
from datetime import timedelta
from src.config import drug_col, lab_cols, demo_col, labdict, labnames, blood, vital, lab_demo_col, ccidict
from multiprocessing import Pool
import multiprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, f1_score

import matplotlib.font_manager as fm
import os
import matplotlib.pyplot as plt

from src.utils import dataload
from src.stat_function import *
from scipy import stats
from src.model import SE_detect

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
tqdm.pandas()

plt.style.use('seaborn')
plt.rcParams['axes.unicode_minus'] = False
# plt.style.use('seaborn')
fe = fm.FontEntry(
    fname=r'../tmp/ArialCE.ttf', # ttf 파일이 저장되어 있는 경로
    name='arial')                       
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 20, 'font.family': 'arial'}) 
torch.set_num_threads(64)

def r2(v):
    return "{:.2f}".format(round(v, 2))
def r3(v):
    return "{:.3f}".format(round(v, 3))
def r4(v):
    return "{:.4f}".format(round(v, 4))


demo_cols = [
    'age', 'sex', 'tumor', 'MI', 'uncomplicated_diabetes',
    'kidney', 'complicated_diabetes', 'HF', 'metastatic', 'dimentia',
    'cerebrovascular', 'peripheral', 'pulmonary', 'liver', 'mild_liver',
    'ulcer', 'hemiplegia', 'tissue', 'HIV', 'anta', 'anti'
]

with open('../usedata/cdiff_timeline.pkl', 'rb') as f:
    cdiff_timeline = pickle.load(f)
with open('../usedata/cdiff_timeline_final.pkl', 'rb') as f:
    tl = pickle.load(f)
with open('../usedata/no_cdiff_timeline.pkl', 'rb') as f:
    no_cdiff_timeline = pickle.load(f)
with open('../usedata/no_cdiff_timeline_final.pkl', 'rb') as f:
    notl = pickle.load(f)
with open('../usedata/tmp/meanstd.pkl', 'rb') as f:
    meanstd = pickle.load(f)
    meanstd = meanstd.loc[lab_cols]

with open('../usedata/cdiff_demo.pkl', 'rb') as f:
    cdiff_demo = pickle.load(f)
with open('../usedata/no_cdiff_demo.pkl', 'rb') as f:
    no_cdiff_demo = pickle.load(f)

def demoset(cdiff_demo, cdiff):
    cdiff_demo.columns = [ccidict[col] if col in ccidict.keys() else col for col in cdiff_demo.columns]
    cdiff_demo['cdiff'] = cdiff
    cdiff_demo = cdiff_demo[cdiff_demo.columns.difference(['cdiff_date', 'index_date', 'index-7', 'index+28', 'visit_source_value'], sort=False)]
    cdiff_demo.reset_index(drop=True, inplace=True)
    return cdiff_demo

cdiff_demo, no_cdiff_demo = demoset(cdiff_demo, 1), demoset(no_cdiff_demo, 0)

class makeargs:
    bs = 256
    hid_dim = 256

args = makeargs

lab_tl = torch.Tensor(np.array([v.loc[lab_cols].astype(float).values for k, v in tl.items()]))
lab_notl = torch.Tensor(np.array([v.loc[lab_cols].astype(float).values for k, v in notl.items()]))
drug_tl = torch.Tensor(np.array([v.loc[drug_col].astype(float).values for k, v in tl.items()]))
drug_notl = torch.Tensor(np.array([v.loc[drug_col].astype(float).values for k, v in notl.items()]))

seed = 64
train_loader, val_loader, test_loader = dataload(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, seed)
for timeline, drugline, demo, label in test_loader: break
input_size, length, demo_length = timeline.shape[1], timeline.shape[-1], demo.shape[-1]


device = 'cuda:5'
torch.set_num_threads(8)
softmax = torch.nn.Softmax()
demo = 0

modelname, args.layers, args.hid_dim = ['gru', 2, 64]
modelpth = f'/workspace/results/saved/{modelname}_{args.layers}_{args.hid_dim}_256_0.0001_0.05_2/best_model_64.tar'
model = SE_detect(modelname, input_size, device, args)
model.load_state_dict(torch.load(modelpth, map_location=device)['model'])
model = model.to(device)

model.eval()
with torch.no_grad():
    vloss1, vloss0 = [], []
    
    loss_sum = 0
    skipcnt = 0
    
    for timeline, drugline, demo, label in tqdm(test_loader):
        index1 = torch.where(label == 1)[0]
        index0 = torch.where(label == 0)[0]
        
        timeline, demo, flag = timeline.to(device), demo.to(device), label.type(torch.long).to(device)
        _vloss = []
        if len(index1) != 0:
            for _t in range(2, 35):
                output1 = model(timeline[index1].permute(0, 2, 1)[:, :_t, :], demo[index1])
                _vloss.append(softmax(output1)[:, 1].cpu().detach().numpy())
            vloss1.append(_vloss)

        _vloss = []
        if len(index0) != 0:
            for _t in range(2, 35):
                output1 = model(timeline[index0].permute(0, 2, 1)[:, :_t, :], demo[index0])
                _vloss.append(softmax(output1)[:, 1].cpu().detach().numpy())
            vloss0.append(_vloss)

vloss1 = np.concatenate([np.transpose(i) for i in vloss1])
vloss0 = np.concatenate([np.transpose(i) for i in vloss0])


fig, ax1 = plt.subplots(figsize=(10.8, 11.445))
ax1.patch.set_facecolor('white')
plt.rcParams['svg.fonttype'] = 'none'

mean1, std1, mean0, std0 = vloss1.mean(axis=0), vloss1.std(axis=0), vloss0.mean(axis=0), vloss0.std(axis=0)
ax1.plot(mean1, label='Infection')
ax1.fill_between(np.arange(33), mean1 - std1, mean1 + std1, alpha=.2)
ax1.plot(mean0, label='Non-infection')
ax1.fill_between(np.arange(33), mean0 - std0, mean0 + std0, alpha=.2)

ax1.grid(color='black', alpha=.1)
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
legend = plt.legend(loc=(0.04, 0.85), frameon='True', fontsize=20)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
ax1.set_xlabel('Days', fontsize=27)
ax1.set_ylabel('Risk Score', fontsize=27)
ax1.set_ylim(0.15, 0.5)
plt.margins(x=0)
plt.savefig('../results/risk_snuh.svg', format='svg')
plt.show()