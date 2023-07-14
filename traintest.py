import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim

from utils.dl_func import EarlyStopping, fcnSaveCheckpoint, fcnLoadCheckpoint
import tqdm

from datetime import datetime as dtt

class TrainTest():
    def __init__(self,  device='gpu:0', metrics=None,
            num_classes=1, model=None, batch_size=1,
            max_epochs=5, criterion=None, optimizer=None, scheduler=None, 
            model_save_path=None, tb_writer=None, save_prefix=None,
            data_loader_train=None, data_loader_val=None,
            data_loader_test=None):
        """
        Initialization
        Inputs:
        data_loader_train - data loader for the training set
        data_loader_val - data loader for the vaidation set
        data_loader_test - data loader for the test set
        model - model fitted
        criterion - loss function
        optimizer - optimizer
        scheduler - scheduler
        metrics - class for metrics which saves everything
        device - gpu or CPU
        params - parameters for the experiment
        tb_writer - tensorboard writer
        """

        super(TrainTest, self).__init__()

        ### do some checks
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val
        self.data_loader_test = data_loader_test
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.scheduler = scheduler
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tb_writer = tb_writer
        self.save_prefix = save_prefix
        self.model_save_path = model_save_path
        self.tqdm_disable = True
        self.metric_save_len = 99999 ##disable
        self.softmax = nn.Softmax(dim=1)


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
        elif phase == 'test':
            ### switch to evaluate
            self.model.eval()
            dataloader = self.data_loader_test

        ### initialize loss for this epoch
        lossTotal = 0.0 #loss

        with torch.set_grad_enabled(phase=='train'):
            ### iterator over batches
            for x, y, path in tqdm.tqdm(dataloader,
                                        desc=f'Epoch {epoch}/{self.max_epochs}, {phase:>10}',
                                        total=len(dataloader),
                                        disable = self.tqdm_disable):
                # # Move the data to the GPU
                x = x.to(self.device)
                y = y.to(self.device)

                ### compute gradient and do SGD step
                self.optimizer.zero_grad()
                
                ###forward + backward + optimize
                yh = self.model(x)
                cLoss = self.criterion(yh, y)
                if phase == 'train':
                    cLoss.backward()
                    self.optimizer.step()
            
                ### convert to probability
                prob = self.softmax(yh)                

               ### measure accuracy and record loss
                lossTotal += cLoss.item()
                
                ### append to output
                self.metrics.add_data(y, prob, path)

        if phase == 'test':
            self.metrics.save_results(save_path=self.model_save_path, 
                                      save_prefix=f'{self.save_prefix}.csv')
        
        mean_loss = lossTotal/len(dataloader)
        auc, auc_std, acc, f1 = self.metrics.calc_metrics(mean_loss)
        
        ### write tesorboard
        if self.tb_writer  is not None:
            self.tb_writer.add_scalar('Loss/{phase}', mean_loss, epoch)
            self.tb_writer.add_scalar('ACC/{phase}', acc, epoch)
            self.tb_writer.add_scalar('ACC_STD/{phase}', auc_std, epoch)
            self.tb_writer.add_scalar('AUC/{phase}', auc, epoch)
            self.tb_writer.add_scalar('F1/{phase}', f1, epoch)
                
        log = {f'{phase}_loss': mean_loss,
               f'{phase}_accuracy': acc,
               f'{phase}_auc': auc,
               f'{phase}_auc_std': auc_std,
               f'{phase}_f1': f1,
               f'{phase}_epoch': epoch}
        print(f'Phase: {phase}, Epoch: {epoch}, Loss: {mean_loss}, ACC: {acc}, F1: {f1}, AUC: {auc}±{auc_std}')
        self.metrics.clear_dict()
        return log

    def fit(self, opt_dir='down', params={}):
        """
        loop through epochs and fits model to data
        Inputs:
        opt_dir - 
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
        
        ### train test
        fcnSaveCheckpoint(model=self.model, optimizer=self.optimizer,
                    epoch=kEpoch, metric_value=save_metric.best_score,
                    filepath=f'{self.model_save_path}/weights/{self.save_prefix}.ckpt',
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
        print('\n' + '#' * 10 + ' Beginning testing ' + '#' * 10)
        
        test_log = self.run('test', kEpoch) #save_output

        train_log.update(test_log)
        train_log.update(val_log)        
        return train_log