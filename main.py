import argparse
import csv
import os
from datetime import datetime

import pandas as pd
import torch
from augmentation.finetuning import EDESTestTransforms2D, EDESTrainTransforms2D
from augmentation.pretraining import SSLTrainTransform2D
from config import configure_loss, configure_optimizers
from data_module import DataGenerator, S_DataGenerator
from evaluation import EvalMetrics
from model.cnn import CNN, EDES_CNN
from ssl_traintest import SSL_TrainTest
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from traintest import TrainTest
from utils.dl_func import load_partial_weights
from utils.func import (duplicate_check, get_input_datasize, get_runname, str_flag)

num_workers = 8
torch.set_num_threads(num_workers)
print('gpu counts', torch.cuda.device_count())

def main(params):
    
    ########################## Common ###########################
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(f"{SAVE_DIR}/{params['classification']}/dl_results.csv"):
        df_done = pd.read_csv(f"{SAVE_DIR}/{params['classification']}/dl_results.csv")
        duplicate_check(df_done, params)
        
    save_path = f"{SAVE_DIR}/{params['classification']}"
    
    params['filename_suffix'] = get_runname(params)
    run_name = params['timestamp']

    if params['save_tensorboard']:
        tb_writer = SummaryWriter(log_dir=f"{SAVE_DIR}/{params['classification']}/tensorboard/{params['filename_suffix']}",
                                  filename_suffix=run_name)
    else:
        tb_writer = None

    ### get DL input size
    transform_params = get_input_datasize(params['dl_model'])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    ### number of classes
    if params['classification'] == 'ccf_cm':
        num_classes = 4
    elif params['classification'] == 'acdc':
        num_classes = 5
    else:
        num_classes = 2048
     
    ########################## Train test mode ###########################
    if params['classification'] in ['ccf_cm', 'acdc']:

        ### Augmentations
        train_transforms = EDESTrainTransforms2D(transform_params)
        test_transforms = EDESTestTransforms2D(transform_params)   
        data_path = f"{DATA_DIR}/{params['dataset_name']}.csv"

        ### Input image
        if 'swapBG'==params['input_img']:
            input_img = 'full'
        else:
            input_img = params['input_img']

        ### Initialize overall dataset
        dataset_train = DataGenerator(
            file_path=data_path, phase='train', input_img=params['input_img'],
            classification=params['classification'], transform=train_transforms
        )
        dataset_test = DataGenerator(
            file_path=data_path, phase='test', input_img=input_img,
            classification=params['classification'], transform=test_transforms
        )
        
        ### Create the dataloader objects for each dataset
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params['batch_size'],
                                            shuffle=True, pin_memory=True,
                                            num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=params['batch_size'],
                                            shuffle=False, pin_memory=True, 
                                            num_workers=num_workers)
        
        ### load model
        model = EDES_CNN(params['dl_model'], params['pt_weight'], num_classes,
                        feature_extract=params['feature_extract'])
        
        ### load pre-trained weight
        if not params['pt_weight'] in ['imagenet', None]:   
            ssl_path = f"{SAVE_DIR}/{params['pt_weight']}"
            df_weight = pd.read_csv(f"{ssl_path}/dl_results.csv")
            df_weight = df_weight.fillna(value='None')
            df_weight = df_weight[(df_weight['dl_model']==params['dl_model']) & 
                                (df_weight['classification']==params['pt_weight'])& 
                                (df_weight['batch_size']==params['pt_batch_size']) & 
                                (df_weight['learning_rate']==params['pt_learning_rate']) & 
                                (df_weight['input_img']==params['pt_input_img']) &
                                (df_weight['dataset_name']==params['pt_dataset_name']) &
                                (df_weight['feature_extract']=='None') &
                                (df_weight['pt_weight']==('None' if params['pt_init_weight'] is None else params['pt_init_weight']))]
            df_weight['weight_path'] = ssl_path + "/weights/" + df_weight['timestamp'].astype(str) + ".ckpt" 
            df_weight = df_weight.sort_values('timestamp', ascending=False)
            print(df_weight.iloc[0].to_dict())
            weight_full_path = df_weight['weight_path'].iloc[0]            
            pretrained_dict = torch.load(weight_full_path, map_location=device)
            load_partial_weights(model, pretrained_dict['state_dict'])
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        
        ### move to GPU    
        model.to(device)

        n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'trainable parameters: {n_train_params}')

        ### initialize EvalMetrics
        metrics = EvalMetrics(num_classes)
        
        ### loss function
        train_labels = dataset_train.df['diagnosis_num']
        criterion = configure_loss(train_labels, device)

        ### optimizer
        optimizer = configure_optimizers(params['opt'], model, params['learning_rate'])

        train_test = TrainTest(
            device=device, metrics=metrics, outcome_var=None,
            num_classes=num_classes, model=model, batch_size=params['batch_size'],
            max_epochs=params['max_epochs'], criterion=criterion, optimizer=optimizer, scheduler=None, 
            model_save_path=save_path, tb_writer=tb_writer, save_prefix=run_name,
            data_loader_train=train_loader, data_loader_val=test_loader,
            data_loader_test=test_loader)            
        dl_log = train_test.fit(params=params)
        
        
    ########################## SSL mode ###########################
    else:
        data_path = f"{DATA_DIR}/{params['dataset_name']}.csv"
        
        ### Augmentations
        train_transforms = SSLTrainTransform2D(transform_params)        
        
        ### Initialize overall dataset
        dataset_train = S_DataGenerator(
            file_path=data_path, phase='train', input_img=params['input_img'],
            classification=['classification'], transform=train_transforms
        )
        
        train_size = int(0.8 * len(dataset_train))
        test_size = len(dataset_train) - train_size
        train_gen, test_gen = random_split(dataset_train, [train_size, test_size],
                                      generator=torch.Generator().manual_seed(123))
        
        ### Create the dataloader objects for each dataset
        train_loader = torch.utils.data.DataLoader(train_gen, batch_size=params['batch_size'],
                                            shuffle=True, pin_memory=True, drop_last=True,
                                            num_workers=num_workers)
        
        test_loader = torch.utils.data.DataLoader(test_gen, batch_size=params['batch_size'],
                                            shuffle=False, pin_memory=True, drop_last=True,
                                            num_workers=num_workers)
    
        ### CNN model
        model = CNN(arch=params['dl_model'], weight_path=params['pt_weight'],  
                    num_classes=num_classes, feature_extract=params['feature_extract'])
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)

        ### move to GPU 
        model.to(device)
        
        ### initialize EvalMetrics
        metrics = EvalMetrics(num_classes)
            
        train_test = SSL_TrainTest(
            device=device, metrics=metrics, outcome_var=None,
            num_classes=num_classes, model=model, batch_size=params['batch_size'],
            max_epochs=params['max_epochs'], scheduler=None, learning_rate=params['learning_rate'],
            model_save_path=save_path, tb_writer=tb_writer, ssl_method=params['classification'],
            data_loader_train=train_loader, data_loader_val=test_loader)    
        dl_log = train_test.fit(save_prefix=run_name, params=params)

    ########################## save results ###########################
    param_csv_file = f"{SAVE_DIR}/{params['classification']}/dl_results.csv"
    params.update(dl_log)
    print(params)
    if os.path.exists(param_csv_file):
        with open(param_csv_file, mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(params.keys()))
            writer.writerow(params)
    else:
        with open(param_csv_file, mode='w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(params.keys()))
            writer.writeheader()
            writer.writerow(params)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('--batch_size', type=int, default=256, 
                        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=200, 
                        help='number of epochs')
    parser.add_argument('--opt', type=str, default='adamw', 
                        help='deep learning optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--classification', type=str, default='amyl_hcm', 
                        help='classification type')
    parser.add_argument('--dl_model', type=str, default= 'DenseNet121',#'ResNet50', 
                        help='Name of CNN model')
    parser.add_argument('--dataset_name', type=str, default='', 
                        help='the name of dataset')
    parser.add_argument('--feature_extract' , type=str_flag, default=None,
                        help='linear evaluation')
    parser.add_argument('--pt_weight', type=str_flag, default='None', 
                        help='path ssl weight')
    parser.add_argument('--save_tensorboard' , type=str_flag, default=True,
                        help='save tensorboard')
    parser.add_argument('--pt_batch_size',  type=int, default=None,
                        help='pre-train batch size')
    parser.add_argument('--pt_learning_rate',  type=float, default=None,
                        help='pre-train learning rate')
    parser.add_argument('--pt_init_weight',  type=str, default=None,
                        help='pre-train initial weight')
    parser.add_argument('--input_img',  type=str, default='full',
                        help='input image style')
    parser.add_argument('--pt_input_img',  type=str, default='full',
                        help=' pre-train input image style')
    parser.add_argument('--pt_dataset_name', type=str, default='', 
                        help='pre-train the name of dataset')
    
    
        
    args = parser.parse_args()

    ### argument to dictionary
    params = vars(args)
    SAVE_DIR = f"/data/aiiih/projects/nakashm2/css_saliency/output"
    DATA_DIR = f"/data/aiiih/projects/nakashm2/css_saliency/data"

    ### print param settings
    print(params)

    params.update({'timestamp':str(datetime.timestamp(datetime.now()))})
    # pre_weights = None
    main(params)
    
