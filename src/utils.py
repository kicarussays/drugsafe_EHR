import os
import numpy as np
import pandas as pd
import random
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import drug_col, lab_cols, demo_col, lab_demo_col


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device


    def forward(self, input, target):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(input, target, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, dtype=torch.float).to(self.device)
            # in case that a minority class is not selected when mini-batch sampling
            if len(self.alpha) != len(torch.unique(target)):
                temp = torch.zeros(len(self.alpha)).to(self.device)
                temp[torch.unique(target)] = alpha.index_select(0, torch.unique(target))
                alpha_t = temp.gather(0, target)
                loss = alpha_t * loss
            else:
                alpha_t = alpha.gather(0, target)
                loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss
    

def seedset(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def timeline_data(timeline):
    drug_col = [
        'Amikacin',
        'Amoxicillin', 'Ampicillin', 'Azithromycin', 'Aztreonam'
        'Cefazolin', 'Cefditoren', 'Cefepime', 'Cefixime', 'Cefotaxime',
        'Cefotetan', 'Cefpodoxime', 'Ceftazidime', 'Ceftriaxone', 'Cefuroxime',
        'Cephradine', 'Cilastatin', 'Ciprofloxacin', 'Clarithromycin',
        'Clavulanate', 'Clindamycin', 'Doripenem', 'Doxycycline', 'Ertapenem',
        'Gentamicin', 'Imipenem', 'Levofloxacin', 'Linezolid', 'Meropenem',
        'Minocycline', 'Moxifloxacin', 'Nafcillin', 'Neomycin',
        'Penicillin', 'Piperacillin', 'Streptomycin', 'Sulbactam',
        'Sulfamethoxazole', 'Teicoplanin', 'Tetracycline', 'Tigecycline',
        'Tobramycin', 'tazobactam', 'trimethoprim'
    ]
    tmp = timeline.loc[lab_cols].sum().astype(bool)
    tmp = tmp.to_frame(name='exist')
    cdiff_date = tmp[tmp['exist']].index.max()
    use_lab_timeline = timeline.loc[lab_demo_col, :cdiff_date]
    _size = use_lab_timeline.shape
    empty_array = np.empty((_size[0], 35-_size[1]))
    empty_array.fill(np.nan)
    filled_timeline = pd.concat([
        pd.DataFrame(empty_array, index=lab_demo_col), 
        use_lab_timeline], axis=1)

    filled_timeline = filled_timeline.astype(float).interpolate(axis=1, limit_direction='both')
    filled_timeline = filled_timeline.fillna(0)
    noise = np.random.normal(0, 0.01, filled_timeline.shape) 
    filled_timeline = filled_timeline + noise

    use_drug_timeline = timeline.loc[drug_col, :cdiff_date]
    _size = use_drug_timeline.shape
    empty_array = np.empty((_size[0], 35-_size[1]))
    empty_array.fill(np.nan)
    drug_filled_timeline = pd.concat([
        pd.DataFrame(empty_array, index=drug_col), 
        use_drug_timeline], axis=1)

    return pd.concat([filled_timeline, drug_filled_timeline.fillna(0)])


def timeline_list(timeline):
    pass


class TL_Dataset(Dataset):
    def __init__(self, lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, idx_1, idx_0):
        self.tl = torch.cat((lab_tl[idx_1], lab_notl[idx_0]), 0)
        self.drug = torch.cat((drug_tl[idx_1], drug_notl[idx_0]), 0)
            
        self.demo = pd.concat([cdiff_demo.iloc[idx_1], no_cdiff_demo.iloc[idx_0]])

        exclude_cols = [col for col in cdiff_demo.columns if 'prev' in col or 'post' in col]
        self.demo = self.demo[cdiff_demo.columns.difference(['person_id', 'labavgday', 'labavgcnt', 'cdiff'] + exclude_cols, sort=False)]
        self.demo['anti'] = (self.demo['anti'] - 2.8) / 1.7
        self.demo = torch.Tensor(self.demo.values).type(torch.float32)
        
        self.label = torch.cat([torch.ones(len(idx_1)), torch.zeros(len(idx_0))])
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # Return Age, Sex, Signal
        return self.tl[idx], self.drug[idx], self.demo[idx], self.label[idx]
    
    
class TL_Dataset(Dataset):
    def __init__(self, lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, idx_1, idx_0):
        self.tl = torch.cat((lab_tl[idx_1], lab_notl[idx_0]), 0)
        self.drug = torch.cat((drug_tl[idx_1], drug_notl[idx_0]), 0)
            
        self.demo = pd.concat([cdiff_demo.iloc[idx_1], no_cdiff_demo.iloc[idx_0]])

        exclude_cols = [col for col in cdiff_demo.columns if 'prev' in col or 'post' in col]
        self.demo = self.demo[cdiff_demo.columns.difference([
            'person_id', 'labavgday', 'labavgcnt', 'cdiff', 'cdiff_date', 'num',
            'visit_source_value', 'index_date', 'index-7', 'index+28'] + exclude_cols, sort=False)]
        self.demo['anti'] = (self.demo['anti'] - 2.8) / 1.7
        self.demo = torch.Tensor(self.demo.values).type(torch.float32)
        
        self.label = torch.cat([torch.ones(len(idx_1)), torch.zeros(len(idx_0))])
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        # Return Age, Sex, Signal
        return self.tl[idx], self.drug[idx], self.demo[idx], self.label[idx]
    
    
def dataload(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, seed):
    idx1, idx0 = list(range(len(lab_tl))), list(range(len(lab_notl)))
    
    train_idx_1, val_idx_1 = train_test_split(idx1, test_size=0.4, random_state=seed)
    train_idx_0, val_idx_0 = train_test_split(idx0, test_size=0.4, random_state=seed)
    val_idx_1, test_idx_1 = train_test_split(val_idx_1, test_size=0.5, random_state=seed)
    val_idx_0, test_idx_0 = train_test_split(val_idx_0, test_size=0.5, random_state=seed)

    train_tlds = TL_Dataset(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, train_idx_1, train_idx_0)
    val_tlds = TL_Dataset(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, val_idx_1, val_idx_0)
    test_tlds = TL_Dataset(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, test_idx_1, test_idx_0)
    train_tldl = DataLoader(train_tlds, batch_size=256, shuffle=True)
    val_tldl = DataLoader(val_tlds, batch_size=256, shuffle=False)
    test_tldl = DataLoader(test_tlds, batch_size=256, shuffle=False)
    
    return train_tldl, val_tldl, test_tldl


def dataload_ml(tl, notl, seed):
    train_idx_1, val_idx_1 = train_test_split(np.array([k for k in tl.keys()]), test_size=0.3, random_state=seed)
    train_idx_0, val_idx_0 = train_test_split(np.array([k for k in notl.keys()]), test_size=0.3, random_state=seed)
    val_idx_1, test_idx_1 = train_test_split(val_idx_1, test_size=0.5, random_state=567)
    val_idx_0, test_idx_0 = train_test_split(val_idx_0, test_size=0.5, random_state=567)
    
    if any(i in np.concatenate([val_idx_1, test_idx_1, val_idx_0, test_idx_0, train_idx_0]) for i in train_idx_1): 
        print('Duplicate Error')
    if any(i in np.concatenate([val_idx_1, test_idx_1, val_idx_0, test_idx_0, train_idx_1]) for i in train_idx_0): 
        print('Duplicate Error')
    
    return train_idx_1, val_idx_1, test_idx_1, train_idx_0, val_idx_0, test_idx_0


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


