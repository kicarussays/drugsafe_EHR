import numpy as np
from numpy import argmax
import os
from tqdm import tqdm
import gc

import torch
import torch.optim as optim
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, f1_score
from src.utils import _ECELoss

torch.set_num_threads(32)

class AdversePrediction:
    def __init__(self, model, train_loader, val_loader, criterion, args, savepath, logger):            
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.savepath = savepath
        self.criterion = criterion
        self.logger = logger
        self.softmax = torch.nn.Softmax(dim=1)
    

    def train(self):
        if self.args.device == 'cpu':
            self.d = 'cpu'
        else:
            self.d = f'cuda:{self.args.device}'
        os.makedirs(self.savepath, exist_ok=True)

        bestauc = 0.0
        patience = 0
        self.model = self.model.to(self.d)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        val_loss, scores = self.evaluation(self.model, self.val_loader)
        roc_auc, pr_auc, sensitivity, specificity, precision, f1score = scores
        self.logger.info('Epoch [0/{}], Val Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                    self.args.max_epoch, val_loss, roc_auc, pr_auc))
        
        first_iter = 0
        # if os.path.isfile(self.savepath + 'best_model.tar'):
        #     print('file exists')
        #     checkpoint = torch.load(self.savepath + 'best_model.tar', map_location=self.d)
        #     self.model.load_state_dict(checkpoint["model"])
        #     self.optimizer.load_state_dict(checkpoint["optimizer"])
        #     first_iter = checkpoint["epoch"] + 1

        for epoch in range(first_iter, self.args.max_epoch):
            self.model.train()
            loss_sum = 0
            skipcnt = 0
            
            for timeline, drugline, demo, label in tqdm(self.train_loader):
                
                if label.shape[0] == 1:
                    skipcnt += 1
                    continue

                timeline, demo, flag = timeline.to(self.d), demo.to(self.d), label.type(torch.long).to(self.d)
                output = self.model(timeline.permute(0, 2, 1), demo)
                loss = self.criterion(output, flag)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_sum += loss.item()

            if (epoch + 1) % self.args.valtime == 0:
                val_loss, scores = self.evaluation(self.model, self.val_loader)
                roc_auc, pr_auc, sensitivity, specificity, precision, f1score = scores
                self.logger.info('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, AUROC: {:.4f}, AUPRC: {:.4f}'.format(
                            epoch+1, self.args.max_epoch, loss_sum / (len(self.train_loader) - skipcnt), val_loss, roc_auc, pr_auc))
                
                if roc_auc > bestauc + 0.001:
                    self.logger.info(f'Saved Best Model...')
                    saving_dict = {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                    }
                    torch.save(saving_dict, self.savepath + f'best_model_{self.args.seed}.tar')
                    bestauc = roc_auc
                    patience = 0
                else:
                    patience += 1

            if self.args.es and patience > self.args.patience:
                self.logger.info(f'Early Stopping Activated')
                break
        
        load_model = torch.load(self.savepath + f'best_model_{self.args.seed}.tar', map_location=self.d)
        del val_loss, scores
        gc.collect()
        torch.cuda.empty_cache()
        
        
        return load_model, bestauc



    def evaluation(self, model, dl):
        model.eval()
        with torch.no_grad():
            vloss, flags = [], []
            
            loss_sum = 0
            # eceloss_sum = 0
            skipcnt = 0
            
            for timeline, drugline, demo, label in dl:
                
                if label.shape[0] == 1:
                    skipcnt += 1
                    continue

                timeline, demo, flag = timeline.to(self.d), demo.to(self.d), label.type(torch.long).to(self.d)
                output = self.model(timeline.permute(0, 2, 1), demo)
                loss = self.criterion(output, flag)
                loss_sum += loss.item()
                # eceloss_sum += _ECELoss(output, flag).item()
                vloss.append(self.softmax(output)[:, 1].view(-1).cpu().detach().numpy())
                flags.append(flag.view(-1).cpu().detach().numpy().astype(int))
            
            vloss = np.concatenate(vloss)
            flags = np.concatenate(flags)

            fpr, tpr, thresholds = roc_curve(flags, vloss)
            precision, recall, _ = precision_recall_curve(flags, vloss)
            roc_auc = auc(fpr, tpr)
            pr_auc =  auc(recall, precision)

            # get the best threshold
            J = tpr - fpr
            ix = argmax(J)
            best_thresh = thresholds[ix]
            y_prob_pred = (vloss >= best_thresh).astype(bool)
            precision = precision_score(flags, y_prob_pred)
            f1score = f1_score(flags, y_prob_pred)
            
            return loss_sum / (len(dl) - skipcnt), (roc_auc, pr_auc, tpr[ix], 1-fpr[ix], precision, f1score)
    
    
