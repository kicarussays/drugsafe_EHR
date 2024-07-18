import argparse
import logging
import traceback
import os
import numpy as np
import pandas as pd
import torch
import pickle
import gc

from src.model import SE_detect
from src.utils import FocalLoss, dataload, seedset
from src.train import AdversePrediction
from src.config import lab_cols, vital, blood, drug_col, ccidict


parser = argparse.ArgumentParser(description='classifier Training')
parser.add_argument('--device', '-d', type=str, 
                    help='cpu or GPU Device Number', default='cpu')
parser.add_argument('--bs', type=int, 
                    help='Batch Size', default=256)
parser.add_argument('--lr', type=float, 
                    help='Learning Rate', default=0.0001)
parser.add_argument('--max-epoch', '-e', type=int, 
                    help='Max Epochs for Each Site', default=1000)
parser.add_argument('--alpha', type=float, 
                    help='alpha of focal loss', default=0.05)
parser.add_argument('--gamma', type=float, 
                    help='gamma of focal loss', default=2)
parser.add_argument('--valtime', '-v', type=int, 
                    help='Validation Interval', default=1)
parser.add_argument('--es', type=int, 
                    help='early stopping option', default=1)
parser.add_argument('--patience', '-p', type=int, 
                    help='Early Stopping Patience', default=20)
parser.add_argument('--model', type=str, 
                    help='rnn, lstm, gru, tf, retain', default='rnn')
parser.add_argument('--layers', type=int, 
                    help='number of layers', default=4)
parser.add_argument('--hid-dim', type=int, 
                    help='hidden dimension', default=256)

args = parser.parse_args()
device = f'cuda:{args.device}'


        
if __name__ == "__main__":
    path = '../usedata/'
    cdiff_pth = os.path.join(path, f'cdiff_timeline_final.pkl')
    no_cdiff_pth = os.path.join(path, f'no_cdiff_timeline_final.pkl')
    with open('../usedata/tmp/all_demo.pkl', 'rb') as f:
        all_demo = pickle.load(f)

    with open(cdiff_pth, 'rb') as f:
        tl = pickle.load(f)
    with open(no_cdiff_pth, 'rb') as f:
        notl = pickle.load(f)
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
    
    lab_tl = torch.Tensor(np.array([v.loc[lab_cols].astype(float).values for k, v in tl.items()]))
    lab_notl = torch.Tensor(np.array([v.loc[lab_cols].astype(float).values for k, v in notl.items()]))
    drug_tl = torch.Tensor(np.array([v.loc[drug_col].astype(float).values for k, v in tl.items()]))
    drug_notl = torch.Tensor(np.array([v.loc[drug_col].astype(float).values for k, v in notl.items()]))

    logpth = f'../results/logs/'
    os.makedirs(logpth, exist_ok=True)
    
    for layer in (1, 2, 3, 4, 5):
        for hid_dim in (32, 64, 128, 256, 512):
            args.layers, args.hid_dim = layer, hid_dim
            params = f'{args.model}_{args.layers}_{args.hid_dim}_{args.bs}_{args.lr}_{args.alpha}_{args.gamma}'
            
            # if os.path.exists(os.path.join(logpth, f'{params}.log')): continue
            savepath = f'../results/saved/{params}/'
            os.makedirs(savepath, exist_ok=True)
            
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(logpth, f'{params}.log'))
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s %(levelname)s:%(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # logging.basicConfig(filename=os.path.join(logpth, f'{params}.log'), 
            #             level=logger.info,
            #             format='%(asctime)s:%(levelname)s:%(message)s')
            # logging.getLogger('matplotlib.font_manager').disabled = True

            scoreset = []
            for seed in range(60, 65):
                args.seed = seed
                try:
                    logger.info(f"\n\nSeed: {seed}\n\n")
                    seedset(seed)
                    train_loader, val_loader, test_loader = dataload(lab_tl, lab_notl, drug_tl, drug_notl, cdiff_demo, no_cdiff_demo, seed)

                    for timeline, drugline, demo, label in train_loader: break
                    input_size, length, demo_length = timeline.shape[1], timeline.shape[-1], demo.shape[-1]
                    model = SE_detect(args.model, input_size, device, args)
                    model.to(device)
                    criterion = FocalLoss(gamma=args.gamma, alpha=[args.alpha, 1-args.alpha], device=device)
                    
                    Trainer = AdversePrediction(
                        model=model, 
                        train_loader=train_loader, 
                        val_loader=val_loader, 
                        criterion=criterion, 
                        args=args, 
                        savepath=savepath,
                        logger=logger
                    )
                    
                    load_model, _ = Trainer.train()
                    model.load_state_dict(load_model['model'])
                    
                    logger.info("Final Test Score\n")
                    loss, scores = Trainer.evaluation(model, test_loader)
                    roc_auc, pr_auc, sensitivity, specificity, precision, f1score = scores
                    logger.info("Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}, Precision: {:.4f}, F1-score: {:.4f}".format(
                        loss, roc_auc, pr_auc, sensitivity, specificity, precision, f1score))
                    scoreset.append([roc_auc, pr_auc, sensitivity, specificity, precision, f1score])

                    del model, Trainer, loss, scores
                    gc.collect()
                    torch.cuda.empty_cache()

                
                except KeyboardInterrupt:
                    import sys
                    sys.exit('KeyboardInterrupt')

                except:
                    logging.error(traceback.format_exc())

            means = np.mean(scoreset, axis=0)
            stds = np.std(scoreset, axis=0)
            logger.info("\n\nAll Stats\n\n")
            logger.info("AUROC: {:.4f} ± {:.4f}".format(means[0], stds[0]))
            logger.info("AUPRC: {:.4f} ± {:.4f}".format(means[1], stds[1]))
            logger.info("Sensitivity: {:.4f} ± {:.4f}".format(means[2], stds[2]))
            logger.info("Specificity: {:.4f} ± {:.4f}".format(means[3], stds[3]))
            logger.info("Precision: {:.4f} ± {:.4f}".format(means[4], stds[4]))
            logger.info("F1-score: {:.4f} ± {:.4f}".format(means[5], stds[5]))

            logger.removeHandler(handler) 
            del logger, handler
            gc.collect()
            torch.cuda.empty_cache()
        