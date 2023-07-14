import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch

sys.path.append('/data/aiiih/projects/nakashm2/css_saliency/code_publish')
from augmentation.finetuning import (EDESTestTransforms2D,
                                     EDESTrainTransforms2D)
from data_module import DataGenerator
from model.cnn import EDES_CNN
from utils.func import (get_input_datasize,
                        str_flag)

from grad_cam import GradCAM

def dice(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    union = np.sum(y_true_f) + np.sum(y_pred_f)
    if union==0: return 1
    intersection = np.sum(y_true_f * y_pred_f)
    return 2. * intersection / union

def ark(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    denominator = np.sum(y_pred_f) + 1e-4
    numerator = np.sum(y_true_f * y_pred_f)
    return numerator/denominator

dic = {}
def get_hook(name):
    def hook(model, input, output):
        dic[name] = output.detach()
    return hook
    
def add_channel(img):
    """ transform (h, w) -> (h, w, c)"""
    img_channel = np.empty((*img.shape, 3))
    img_channel[:,:,0] = img
    img_channel[:,:,1] = img
    img_channel[:,:,2] = img

    return img_channel

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

def save_original(filename, raw_image):
    if raw_image.shape[0] in [2,3]:
        img = add_channel(raw_image.detach().cpu().numpy()[0])
    else:
        img = add_channel(raw_image.detach().cpu().numpy())
    c = (255*(img - np.min(img))/np.ptp(img)).astype(int)        
    cv2.imwrite(filename, np.uint8(c))

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if raw_image.shape[0] in [2,3]:
        img = add_channel(raw_image.detach().cpu().numpy()[0])
    else:
        img = add_channel(raw_image.detach().cpu().numpy())
    c = (255*(img - np.min(img))/np.ptp(img)).astype(int)        
    gcam = (cmap.astype(np.float) + c.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='batch size')
    parser.add_argument('--max_epochs', type=int, default=30, 
                        help='number of epochs')
    parser.add_argument('--opt', type=str, default='adamw', 
                        help='deep learning optimizer')
    parser.add_argument('--learning_rate', type=float, default=0.0001, 
                        help='learning rate')
    parser.add_argument('--classification', type=str, default='acdc', 
                        help='classification type')
    parser.add_argument('--dl_model', type=str, default= 'DenseNet121',#'ResNet50', 
                        help='Name of CNN model')
    parser.add_argument('--dataset_name', type=str, default='acdc', 
                        help='the name of dataset')
    parser.add_argument('--feature_extract' , type=str_flag, default=True,
                        help='linear evaluation')
    parser.add_argument('--pt_weight', type=str_flag, default='imagenet', 
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
    SAVE_DIR = f"/data/aiiih/projects/nakashm2/css_saliency/output3"
    DATA_DIR = f"/data/aiiih/projects/nakashm2/css_saliency/data"
    data_path = f"{DATA_DIR}/{params['dataset_name']}.csv"

    num_workers = 8
    device = 'cpu'
    df =  pd.read_csv(f"/data/aiiih/projects/nakashm2/css_saliency/output/{params['classification']}/dl_results.csv") 
    df = df.fillna(value='None')
    df2 = df[(df['dl_model'] == params['dl_model']) &
            (df['opt']== params['opt']) &
            (df['batch_size']== params['batch_size']) &
            (df['max_epochs']== params['max_epochs']) &
            (df['dataset_name']== params['dataset_name']) &
            (df['classification']== params['classification']) &
            (df['learning_rate']== params['learning_rate']) &
            (df['pt_dataset_name']== ('None' if params['pt_dataset_name'] == '' else params['pt_dataset_name'])) &
            (df['pt_weight']== ('None' if params['pt_weight'] is None else params['pt_weight'])) &
            (df['pt_init_weight']== ('None' if params['pt_init_weight'] is None else params['pt_init_weight'])) &
            (df['pt_batch_size']== ('None' if params['pt_batch_size'] is None else params['pt_batch_size'])) &
            (df['pt_learning_rate']== ('None' if params['pt_learning_rate'] is None else params['pt_learning_rate'])) &
            (df['input_img']== ('None' if params['input_img'] is None else params['input_img'])) &
            (df['pt_input_img']== ('None' if params['pt_input_img'] is None else params['pt_input_img'])) &
            (df['feature_extract']== ('None' if params['feature_extract'] is None else params['feature_extract']))]

    if len(df2) != 1:
        df2 = df2.drop_duplicates('dl_model')

    ### get DL input size
    transform_params = get_input_datasize(params['dl_model'])

    ### number of classes
    if params['classification'] == 'ccf_cm':
        inv_labels = {0:'icm', 1:'nicm', 2:'amyl', 3:'hcm'}
    elif params['classification'] == 'acdc':
        inv_labels = {0:'DCM', 1:'HCM', 2:'MINF', 3:'NOR', 4:'RV'}
    num_classes = len(inv_labels)

    ### linear or finetune model
    if params['feature_extract']:
        phase = 'linear'
    else:
        phase = 'finetune'
        
    if 'swapBG'==params['input_img']:
        input_img = 'full'
    else:
        input_img = params['input_img']

    experiment_name = f"{params['dl_model']}_" \
                        + f"{params['batch_size']}_" \
                        + f"{params['learning_rate']}_" \
                        + f"{params['classification']}_" \
                        + f"{params['dataset_name']}_" \
                        + f"{params['input_img']}_" \
                        + f"{params['pt_init_weight']}P_" \
                        + f"{params['pt_weight']}P_" \
                        + f"{params['pt_batch_size']}P_" \
                        + f"{params['pt_learning_rate']}P_" \
                        + f"{params['pt_input_img']}P_" \
                        + f"{params['pt_dataset_name']}P"
    ### augmentations
    train_transforms = EDESTrainTransforms2D(transform_params)
    test_transforms = EDESTestTransforms2D(transform_params)   

    ### data loader
    dataset_test = DataGenerator(
        file_path=data_path, phase='test', input_img=input_img,
        classification=params['classification'], transform=test_transforms
    )
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=params['batch_size'],
                                            shuffle=False, pin_memory=True,
                                            num_workers=num_workers)

    ### load model
    model = EDES_CNN(params['dl_model'], params['pt_weight'], num_classes,
                    feature_extract=params['feature_extract'])
    ### load weights
    weight_path = f"/data/aiiih/projects/nakashm2/css_saliency/output/{params['classification']}"
    df2['weight_path'] = weight_path + "/weights/" + df['timestamp'].astype(str) + ".ckpt" 
    predf_weight = df2.sort_values('timestamp', ascending=False)
    weight_full_path = df2['weight_path'].iloc[0]            
    pretrained_dict = torch.load(weight_full_path, map_location=device)
    model.load_state_dict(pretrained_dict['state_dict'])
    print(df2.iloc[0].to_dict())

    ### move to GPU    
    model.to(device)
    model.eval()

    ### target layers for visualization
    if params['dl_model']=='VGG16':
        target_layer = 'classifier.encoder.classifier[6]'
        target_layer2 = 'classifier.encoder.features.29'
        model.encoder.classifier[6].register_forward_hook(get_hook(target_layer))
    elif params['dl_model']=='DenseNet121':
        target_layer = 'encoder.classifier'
        target_layer2 = 'encoder.features.norm5'
        model.encoder.classifier.register_forward_hook(get_hook(target_layer))
        
    feature_lst, y_lst =  [], []
    lst_dcs = []
    gcam = GradCAM(model=model)

    for X, y, z in dataset_test:
        X = X[None, :]
        images = X    
        probs = gcam.forward(images)

        ### latent feature
        feature_lst.append(dic[target_layer])
        y_lst.append(int(y))    

        gcam.backward(ids=torch.tensor([[torch.argmax(probs)]]).to(device))
        regions = gcam.generate(target_layer=target_layer2)    

        accession = int(dataset_test.df[dataset_test.df['es_img_path']==z]['AccessionNumber'].iloc[0])
        diagnosis = inv_labels[int(y)]
        
        # Grad-CAM
        save_dir = Path(f"{SAVE_DIR}/{params['classification']}/{phase}/gradcam/{experiment_name}/{diagnosis}/")
        save_dir.mkdir(exist_ok=True, parents=True)
        save_gradcam(
            filename=str(save_dir / f"{accession}-{params['dl_model']}.png"),
            gcam=regions[0, 0],
            raw_image=images[0][0],
        )
        
        # Original
        save_dir = Path(f"{SAVE_DIR}/{params['classification']}/{phase}/original_{params['pt_input_img']}/{diagnosis}/")
        save_dir.mkdir(exist_ok=True, parents=True)
        if not os.path.exists(str(save_dir / f"{accession}-{params['dl_model']}.png")):
            save_original(
                filename=str(save_dir / f"{accession}-{params['dl_model']}.png"),
                raw_image=images[0][0]
            )
        
        ### Only Grad-CAM
        save_dir = Path(f"{SAVE_DIR}/{params['classification']}/{phase}/only_heatmap/{experiment_name}/{diagnosis}/")
        save_dir.mkdir(exist_ok=True, parents=True)
        cam = regions[0, 0]
        np.save(str(save_dir / f"{accession}-{params['dl_model']}.npy"),
                cam.detach().cpu().numpy())
        seg_path = z.replace('/img/', '/seg/')
        seg = np.load(seg_path)
        seg[seg!=0] = 1
        cam = cam.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()

        d_dic = {
            'AccessionNumber':accession,
            'model': params['dl_model'], 
            'experiment_name': experiment_name,
            'classification': params['classification'],
            'phase': phase,  
            'path_heatmap': str(save_dir / f"{accession}-{params['dl_model']}.npy"),
            'seg_path': seg_path,
            'diagnosis':diagnosis, 
            'diagnosis_num':int(y), 
            'probs': probs
        }
        
        for idx in range(len(probs[0])):
            d_dic.update({f'probs_{inv_labels[idx]}': probs[0][idx]})
        
        try:
            dice_score = dice(seg, cam)
            ark_score = ark(seg, cam)
        except:
            dice_score = 0
            ark_score = 0
        d_dic.update({'dice': dice_score, 'ark': ark_score})
        print(d_dic)
        lst_dcs.append(d_dic)
        
    df_dsc = pd.DataFrame(lst_dcs)
    df_dsc.to_csv(f"{SAVE_DIR}/{params['classification']}/{phase}/gradcam/{experiment_name}/{params['dl_model']}.csv", index=False)
