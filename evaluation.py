import os

import numpy as np
import pandas as pd
from utils.func import roc_auc_ci
from sklearn.metrics import accuracy_score, f1_score

class EvalMetrics():
    """
    Class for various evaluation metrics
    """
    def __init__(self, num_classes):
        """
        Initialization
        Input:
        num_classes: number of classes
        """
        self.nLabels = num_classes
        self.x = None
        self.y = None
        self.log_dict = {'y': [], 'yh': [], 'path': []}

    def clear_dict(self):
        self.log_dict = {'y': [], 'yh': [], 'path': []}

    def add_data(self, y, yh, path):
        """adds data to the data pool"""
        self.log_dict['y'].extend(y.detach().cpu().numpy())
        self.log_dict['yh'].extend(yh.detach().cpu().numpy())
        self.log_dict['path'].extend(path)

    def save_results(self,save_path=None, save_prefix=None):
        """save result to the csv files"""
        df_pred = pd.DataFrame(self.log_dict['yh'])
        df_pred['path'] = self.log_dict['path']
        df_pred['accession'] = df_pred['path'].str.split('/').str[-1].str.split('_').str[0].str.split('.npy').str[0]
        df_pred['true_label_num'] = np.array(self.log_dict['y'])
        df_pred['pred_num'] =  np.argmax(self.log_dict['yh'], axis=1)
        if not os.path.exists(f'{save_path}/results'):
            os.makedirs(f'{save_path}/results')
        df_pred.to_csv(f'{save_path}/results/{save_prefix}', index=False)

    def calc_metrics(self, loss):
        """Calc metrics"""
        yh = np.array(self.log_dict['yh'])
        y = np.array(self.log_dict['y'])
        preds = np.argmax(self.log_dict['yh'], axis=1)
        acc = accuracy_score(self.log_dict['y'], preds)
        f1 = f1_score(self.log_dict['y'], preds, average='macro')

        AUC_sum, std_sum = 0, 0
        for idx in range(yh.shape[1]):
            y_true = (y==idx)*1
            y_score =yh[:,idx]
            AUC, std = roc_auc_ci(y_true, y_score, positive=1)
            AUC_sum += AUC
            std_sum += std
        auc = AUC_sum/yh.shape[1]
        auc_std = std_sum/yh.shape[1]
        return auc, auc_std, acc, f1

