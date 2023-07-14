import os
from datetime import datetime as dtt

import pandas as pd
import torch
import tqdm

from ssl_method.byol import BYOL
from ssl_method.moco import Moco_v2
from ssl_method.simclr import SimCLR
from utils.dl_func import EarlyStopping, fcnSaveCheckpoint


class SSL_TrainTest():
    def __init__(self, device, metrics, outcome_var=None,
            num_classes=1, print_freq=1, model=None, batch_size=1,
            max_epochs=5, criterion=None, scheduler=None, learning_rate=1e-5,
            model_save_path='', tb_writer=None,  ssl_method='simclr',
            data_loader_train=None, data_loader_val=None):
        """
        Initialization
        Inputs:
        data_loader_train - data loader for the training set
        data_loader_val - data loader for the vaidation set
        model - model fitted
        criterion - loss function
        scheduler - scheduler
        metrics - class for metrics which saves everything
        device - gpu or CPU
        params - parameters for the experiment
        tb_writer - tensorboard writer
        """
        ### initialize stuff
        super(SSL_TrainTest, self).__init__()

        ### do some checks
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.scheduler = scheduler
        self.device = device
        self.outcome_var = outcome_var
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.print_freq = print_freq
        self.tb_writer = tb_writer
        self.ssl_method = ssl_method
        self.model_save_path = model_save_path
        self.learning_rate = learning_rate
        self.tqdm_disable = True
    
        self.metric_save_len = 99999 ##disable

        if self.ssl_method == 'simclr':
            self.ssl = SimCLR(self.model, hidden_dim=2048, out_dim=128,
                          temperature=0.1, learning_rate=learning_rate, weight_decay=1e-6)
        elif self.ssl_method == 'moco':             
            self.ssl = Moco_v2(self.model, emb_dim=2048, num_negatives=1024,
                               encoder_momentum=0.999, softmax_temperature=0.07,learning_rate=learning_rate,
                               momentum=0.9, weight_decay=1e-4, use_mlp=True)        
        elif self.ssl_method == 'byol':
            self.ssl = BYOL(self.model, learning_rate=learning_rate, weight_decay=1e-6, moving_average_decay = 0.99,
                            encoder_out_dim=2048, projector_hidden_dim=4096, projector_out_dim=128)

        self.ssl.to(device)
        self.optimizer = self.ssl.optimizer

    def run(self, phase, epoch):
        """Trains, validate or test model"""
        if phase == 'train':
            ##### switch to train
            self.model.train()
            dataloader = self.data_loader_train
        elif phase == 'val':
            ### switch to evaluate
            self.model.eval()
            dataloader = self.data_loader_val
        
        ### initialize loss for this epoch
        lossTotal = 0.0 #loss

        with torch.set_grad_enabled(phase == 'train'):
            ### iterator over batches
            for x, y, _ in tqdm.tqdm(dataloader,
                                     desc=f'Epoch {epoch}/{self.max_epochs}, {phase:>10}',
                                     total=len(dataloader),
                                     disable = self.tqdm_disable):
                # Move the data to the GPU
                x1 = x[0].to(self.device)
                x2 = x[1].to(self.device)
                ### compute gradient and do SGD step
                self.optimizer.zero_grad()

                ###forward + backward + optimize
                if self.ssl_method in ['simclr',]:
                    cLoss = self.ssl(x1, x2)
                elif self.ssl_method in ['moco', 'byol']:
                    cLoss, log = self.ssl(x1, x2)

                if phase == 'train':
                    cLoss.backward()
                    self.optimizer.step()

                ### measure accuracy and record loss
                lossTotal += cLoss.item()
                del cLoss
        
        mean_loss = lossTotal/len(dataloader)
        
        ### write to output
        if self.tb_writer  is not None:
            self.tb_writer.add_scalar('Loss/{phase}', mean_loss, epoch)

        log = {f'{phase}_loss': mean_loss, 'epoch':epoch}
        print(f'Phase: {phase}, Epoch: {epoch}, Loss: {mean_loss}')

        self.metrics.clear_dict()
        return log
        #end of batch loop
    #end of train_supervised

    def fit(self, best_metric=None, opt_dir='down', save_prefix=None, params={}):
        """
        loop through epochs and fits model to data
        Inputs:
        best_metric - 
        opt_dir - 'up' or 'down'
        save_prefix - prefix name
        params - dictionary
        """
        ########################## training the model ##########################
        ### initialize model saving
        save_metric = EarlyStopping(patience=self.metric_save_len)

        print(f"start training model at {str(dtt.now().strftime('%H:%M:%S'))}")
        log_dic ={'train_log': [], 'val_log': []}

        ### train model and evaluate after every epoch
        for kEpoch in range(1, self.max_epochs+1):

            ### training
            train_log = self.run('train', kEpoch)

            ### evaluate
            val_log = self.run('val', kEpoch)
            
            log_dic['train_log'].append(train_log)
            log_dic['val_log'].append(val_log)           
            
        fcnSaveCheckpoint(model=self.model, optimizer=self.optimizer,
                epoch=kEpoch, metric_value=save_metric.best_score,
                filepath=f'{self.model_save_path}/weights/{save_prefix}.ckpt',
                params=params)
        ### Save records
        save_record_dir = f'{self.model_save_path}/train_records'
        if not os.path.exists(save_record_dir):
            os.mkdir(save_record_dir)
        train_df = pd.DataFrame(log_dic['train_log'])
        val_df =pd.DataFrame(log_dic['val_log'])
        df = pd.concat([train_df, val_df], axis=1)
        df.to_csv(f"{save_record_dir}/{params['timestamp']}.csv", index=False)
        
        ### end of training
        print(f"end training model at {str(dtt.now().strftime('%H:%M:%S'))}")
        
        train_log.update(val_log)
        return train_log
