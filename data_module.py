import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    """Generates data for pytorch"""
    def __init__(
            self, file_path, phase, input_img,
            classification, transform=None, 
        ):
        """
        Initialization
        Input:
            file_path: path to the csv file
            phase: train, test val phase (ex 'train') 
            input_img: input image type (ex swapBG, seg, BG, full) 
            classification
            transform: augmentation
        """
        self.phase = phase
        self.file_path = file_path
        self.transform = transform 
        self.classification = classification
        self.input_img = input_img
        self.df = self.get_df()
       
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.df)

    def __getitem__(self, index):
        """Generate one batch of data"""
        
        ### get y value
        if self.classification in ['ccf_cm', 'acdc']:
            y = self.df.iloc[index]['diagnosis_num']
        else:
            raise Exception('Something is wrong')
        
        X_es_img = np.load(self.df.iloc[index]['es_img_path'])
        X_es_seg = np.load(self.df.iloc[index]['es_seg_path'])
        
        X_ed_img = np.load(self.df.iloc[index]['ed_img_path'])
        X_ed_seg = np.load(self.df.iloc[index]['ed_seg_path'])
        
        
        if self.input_img == 'swapBG':
            r_index1 = random.randint(0, len(self.df)-1)
            r_index2 = random.randint(0, len(self.df)-1)
            
            filenameBG_temp1 = self.df.iloc[r_index1]['es_img_path']
            filenameBG_temp2 = self.df.iloc[r_index2]['ed_img_path']
            
            X_img1 = np.load(filenameBG_temp1)  
            X_img2 = np.load(filenameBG_temp2)  
        
            X_es_img[X_es_seg==0]= X_img1[X_es_seg==0]
            X_ed_img[X_ed_seg==0]= X_img2[X_ed_seg==0]
        
        elif self.input_img == 'seg':
            X_es_img[X_es_seg==0]= 0
            X_ed_img[X_ed_seg==0]= 0
        
        elif self.input_img == 'BG':
            X_es_img[X_es_seg!=0]= 0
            X_ed_img[X_ed_seg!=0]= 0
        
        ### transform data dimmentions
        X_es_img = np.expand_dims(X_es_img, axis=-1)
        X_ed_img = np.expand_dims(X_ed_img, axis=-1)
        
        ### Augmentation
        X_es_img, X_ed_img = self.transform(X_es_img, X_ed_img)

        X = torch.stack([X_es_img, X_ed_img], 1)
        y = torch.tensor(int(y), dtype=torch.long)        

        return X, y, self.df.iloc[index]['es_img_path']

    def get_df(self):
        df = pd.read_csv(self.file_path)
        ### select phase
        df = df[df['train/test']==self.phase]
        df = df.drop_duplicates(['AccessionNumber'])
        df = df.reset_index()
        return df



class S_DataGenerator(Dataset):
    """Generates data for pytorch"""
    def __init__(
            self, file_path, phase, input_img,
            classification, transform=None
        ):
        """
        Initialization
        Input:
            file_path: path to the csv file
            phase: train, test val phase (ex 'train') 
            input_img: input image type (ex swapBG, seg, BG, full) 
            classification: classification name
            transform: augmentation
        """
        self.phase = phase
        self.file_path = file_path
        self.transform = transform 
        self.classification = classification
        self.input_img = input_img
        self.df = self.get_df()
       
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.df)

    def __getitem__(self, index):
        """Generate one batch of data"""
        img_temp = self.df.iloc[index]['img_path']
        X_img = np.load(img_temp)
        
        if self.input_img == 'swapBG':
            seg_temp = self.df.iloc[index]['seg_path']
            X_seg = np.load(seg_temp)
            
            # randomly select background
            r_index = random.randint(0, len(self.df)-1)
            filenameBG_temp = self.df.iloc[r_index]['img_path']
            X_img2 = np.load(filenameBG_temp)  
            
            # swapping background
            X_img[X_seg==0]= X_img2[X_seg==0]
            
        elif self.input_img == 'seg':
            seg_temp = self.df.iloc[index]['seg_path']
            X_seg = np.load(seg_temp)
            X_img[X_seg==0]= 0
        
        ### get y value
        if self.classification in ['ccf_cm', 'acdc']:
            raise Exception('Something is wrong')
        else:
            y= torch.tensor(0, dtype=torch.long)
                
        ### transform data dimmentions
        X_img = np.expand_dims(X_img, axis=-1)
        
        ### augmentation
        X_img = self.transform(X_img)
        return X_img, y, img_temp

    def get_df(self):
        df = pd.read_csv(self.file_path)
        ### select phase
        df = df[df['train/test']==self.phase]
        df = df.reset_index()
        return df

