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

import shap
from src.model import SE_detect_forshap


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

##### BEGINNNING OF SHAP SCORE CALCULATION

lab_tl = torch.Tensor(np.array([v.loc[lab_cols].astype(float).values for k, v in tl.items()]))
lab_notl = torch.Tensor(np.array([v.loc[lab_cols].astype(float).values for k, v in notl.items()]))
drug_tl = torch.Tensor(np.array([v.loc[drug_col].astype(float).values for k, v in tl.items()]))
drug_notl = torch.Tensor(np.array([v.loc[drug_col].astype(float).values for k, v in notl.items()]))

seed = 64
train_loader, val_loader, test_loader = dataload(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, seed)
for timeline, drugline, demo, label in test_loader: break
input_size, length, demo_length = timeline.shape[1], timeline.shape[-1], demo.shape[-1]

device = 'cuda:2'
softmax = torch.nn.Softmax(dim=1)

modelname, args.layers, args.hid_dim = ['gru', 2, 64]
modelpth = f'/workspace/results/saved/{modelname}_{args.layers}_{args.hid_dim}_256_0.0001_0.05_2/best_model_64.tar'
model = SE_detect_forshap(modelname, input_size, device, args, deep=True)
model.load_state_dict(torch.load(modelpth, map_location=device)['model'])
model.eval()

shaptl, shapdemo = [], []
with torch.no_grad():
    vloss, flags = [], []
    
    loss_sum = 0
    skipcnt = 0
    
    for timeline, drugline, demo, label in tqdm(test_loader):
        shaptl.append(timeline.permute(0, 2, 1).detach().numpy())
        shapdemo.append(demo.detach().numpy())
        
shaptl = np.concatenate(shaptl)
shapdemo = np.concatenate(shapdemo)

shapinput = np.concatenate([shaptl.reshape(shaptl.shape[0], -1), shapdemo], axis=1)

np.random.seed(seed)
idxs = np.random.choice(np.arange(len(shapinput)), size=1000, replace=False)
explainer = shap.DeepExplainer(model, torch.Tensor(shapinput[idxs]))
shap_values = explainer.shap_values(torch.Tensor(shapinput[:94]))
with open('../results/shap_values.pkl', 'wb') as f:
    pickle.dump(shap_values, f, pickle.HIGHEST_PROTOCOL)


bdlab = pd.read_csv('/workspace/results/bd/lab_shap.csv')
bddemo = pd.read_csv('/workspace/results/bd/demo_shap.csv')
with open('../results/shap_values.pkl', 'rb') as f:
    shap_values = pickle.load(f)
max_val = np.max(np.array([np.abs(shap_values[i]) for i in range(len(shap_values))]), axis=0)
score_tl = max_val[:, :-21].reshape(max_val.shape[0], -1, 23)
score_demo = max_val[:, -21:]
plt.rcParams['svg.fonttype'] = 'none'

_lab_cols = [
    'SBP', 'DBP', 'Heart rate', 'Respiratory rate', 'Body temperature', 'WBC', 'Hemoglobin',
    'Platelet', 'Neutrophil', 'ANC', 'Albumin', 'Total protein', 'Total bilirubin', 'AST',
    'ALP', 'ALT', 'BUN', 'Creatinine', 'Sodium', 'Potassium', 'Chloride', 'Total CO2', 'CRP'
]
_demo_cols = [
    'Age', 'Sex', 'Malignant tumor', 'Myocardial infarction', 'Uncomplicated diabetes',
    'Complicated diabetes', 'Renal disease', 'Heart Failure', 'Metastatic carcinoma',
    'Dementia', 'Cerebrovascular disease', 'Peripheral vascular disease', 'Pulmonary disease',
    'Liver disease', 'Mild liver disease', 'Peptic ulcer disease', 'Paraplegia and hemiplegia',
    'Connective tissue disease', 'HIV', 'Number of antibiotics', 'Antacid usage'

]


fig, ax1 = plt.subplots(figsize=(20, 4))
x = np.arange(len(lab_cols))
ax1.patch.set_facecolor('white')
ax1.grid(color='black', alpha=.1)
ax1.bar(x - 0.2, np.mean(np.sum(score_tl, axis=1), axis=0), width=0.4, label='SNUH')
ax1.bar(x + 0.2, bdlab.iloc[:, -1], width=0.4, label='SNUBH')
ax1.set_xticks(x, _lab_cols)
ax1.tick_params(axis='x', rotation=90, labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.set_ylim([0, 0.06])
ax1.set_xlabel('Laboratory tests', fontsize=23)
ax1.set_ylabel('SHAP values', fontsize=23)
legend = plt.legend(loc=(0.02, 0.65), frameon='True', fontsize=20)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
fig.autofmt_xdate(rotation=45)
plt.margins(x=.02)
plt.savefig('../results/labshap.svg', format='svg')
plt.show()


fig, ax1 = plt.subplots(figsize=(20, 4))
x = np.arange(len(demo_cols))
ax1.patch.set_facecolor('white')
ax1.grid(color='black', alpha=.1)
ax1.bar(x - 0.2, np.mean(score_demo, axis=(0)), width=0.4, label='SNUH')
ax1.bar(x + 0.2, bddemo.iloc[:, -1], width=0.4, label='SNUBH')
ax1.set_xticks(x, _demo_cols)
ax1.tick_params(axis='x', rotation=90, labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax1.set_ylim([0, 0.06])
ax1.set_xlabel('Patient information', fontsize=23)
ax1.set_ylabel('SHAP values', fontsize=23)
legend = plt.legend(loc=(0.02, 0.65), frameon='True', fontsize=20)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
fig.autofmt_xdate(rotation=45)
plt.margins(x=.02)
plt.savefig('../results/demoshap.svg', format='svg')
plt.show()