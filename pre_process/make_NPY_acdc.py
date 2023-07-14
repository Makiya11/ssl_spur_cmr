import glob
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

from image_processing_cvi.img_prep import norm, resize_image

seed(123)

def fcnLoadNifti(file_name):
    """
    Use the package nibabel to load a nifti file
    """
    nifti = nib.load(file_name)
    scan = np.asanyarray(nifti.dataobj).astype(np.float32)
    header = nifti.header
    affine = nifti.affine
    return scan, affine, header

def norm_transform_type(img):
    if save_int:
        img[img != 0] = norm(img[img != 0])*255
        img = img.astype('uint8')
    else:
        img[img != 0] = norm(img[img != 0])
        img = img.astype('float32')
    return img



##################################################

ACDC_PATH = '/data/aiiih/data/open/acdc'
SAVE_PATH = '/data/aiiih/projects/nakashm2/css_saliency/data/acdc'
nX = 224
nY = 224
new_pixel_space = 1.5
save_int = True
lst = []
##################################################

for train_test in ['training', 'testing']:
    for pt_dir in glob.glob(f'{ACDC_PATH}/{train_test}/patient*'):        
        info = f'{pt_dir}/Info.cfg'
        
        pt_num = os.path.basename(pt_dir)
        df = pd.read_csv(info ,sep=':', header=None, index_col=0)

        es_phase = "{0:02d}".format(int(df.T['ES'].iloc[0]))
        ed_phase = "{0:02d}".format(int(df.T['ED'].iloc[0]))
        
        es_scan_name = f'{pt_dir}/{pt_num}_frame{es_phase}.nii.gz'
        ed_scan_name = f'{pt_dir}/{pt_num}_frame{ed_phase}.nii.gz'
        
        es_img, _, header_es = fcnLoadNifti(es_scan_name)
        ed_img, _, header_ed = fcnLoadNifti(ed_scan_name)
        
        es_gt_img, _, _ = fcnLoadNifti(str(es_scan_name).replace('.nii.gz', '_gt.nii.gz'))
        ed_gt_img, _, _ = fcnLoadNifti(str(ed_scan_name).replace('.nii.gz', '_gt.nii.gz'))

        save_dir = f"{SAVE_PATH}/{train_test}/{df.T['Group'].iloc[0].replace(' ', '')}"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        ### Pixel norm
        pixel_space_X = header_es['pixdim'][1]
        pixel_space_Y = header_es['pixdim'][2]

        if header_es['pixdim'][1]!=header_es['pixdim'][2]:
            breakpoint()
        if es_img.shape!=ed_img.shape:
            breakpoint()
        
        print(es_img.shape)
        
        resize_factor_X = (pixel_space_X/new_pixel_space)
        resize_factor_Y = (pixel_space_Y/new_pixel_space)
        
        new_shape_X = (es_img.shape[0]*resize_factor_X).round()
        new_shape_Y = (es_img.shape[1]*resize_factor_Y).round()
        
        zoom_factor_X = new_shape_X / es_img.shape[0]
        zoom_factor_Y = new_shape_Y / es_img.shape[1]
        zoom_factor = (zoom_factor_X, zoom_factor_Y)
        
        es_img2 = np.zeros((nX, nY, es_img.shape[2]))
        es_gt_img2 = np.zeros((nX, nY, es_img.shape[2]))
        ed_img2 = np.zeros((nX, nY, ed_img.shape[2]))
        ed_gt_img2 = np.zeros((nX, nY, ed_img.shape[2]))
        
        for idx_slice in range(es_img.shape[2]):
            es_img2[:,:,idx_slice] = resize_image(es_img[:,:, idx_slice], 
                                                new_pixel_space, 
                                                zoom_factor=zoom_factor, 
                                                nDimLimit=(nX,nY), ord=3)
            es_gt_img2[:,:,idx_slice] = resize_image(es_gt_img[:,:,idx_slice], 
                                                    new_pixel_space, 
                                                    zoom_factor=zoom_factor, 
                                                    nDimLimit=(nX,nY), ord=0)
        
        for idx_slice in range(ed_img.shape[2]):
            ed_img2[:,:,idx_slice] = resize_image(ed_img[:,:, idx_slice], 
                                                new_pixel_space, 
                                                zoom_factor=zoom_factor, 
                                                nDimLimit=(nX,nY), ord=3)
            ed_gt_img2[:,:,idx_slice] = resize_image(ed_gt_img[:,:, idx_slice], 
                                                    new_pixel_space, 
                                                    zoom_factor=zoom_factor, 
                                                    nDimLimit=(nX,nY), ord=0)
        es_img = es_img2
        es_gt_img = es_gt_img2
        ed_img = ed_img2
        ed_gt_img = ed_gt_img2
        
        ### Normalize the resolution
        es_img = norm_transform_type(es_img)
        ed_img = norm_transform_type(ed_img)
        
        ### get middle phase index
        mid_slice = int(es_img.shape[2]//2)

        os.makedirs(f'{save_dir}/img', exist_ok=True)
        os.makedirs(f'{save_dir}/seg', exist_ok=True)
        
        ## save npy
        np.save(f'{save_dir}/img/{pt_num}_es.npy', es_img[:,:, mid_slice])  
        np.save(f'{save_dir}/seg/{pt_num}_es.npy', es_gt_img[:,:, mid_slice])  

        np.save(f'{save_dir}/img/{pt_num}_ed.npy', ed_img[:,:, mid_slice])         
        np.save(f'{save_dir}/seg/{pt_num}_ed.npy', ed_gt_img[:,:, mid_slice]) 
        
        dic = {
            'AccessionNumber': int(re.search(r'\d+', pt_num).group()),
            'es_img_path': f'{save_dir}/img/{pt_num}_es.npy',
            'es_seg_path': f'{save_dir}/seg/{pt_num}_es.npy',
            'ed_img_path': f'{save_dir}/img/{pt_num}_ed.npy',
            'ed_seg_path': f'{save_dir}/seg/{pt_num}_ed.npy',
            'diagnosis': df.T['Group'].iloc[0].replace(' ', ''),
            'train/test': train_test.replace('training','train').replace('testing','test')
        }        
        
        lst.append(dic)
df = pd.DataFrame(lst)

df['diagnosis_num'] = df['diagnosis'].map({'DCM':0, 'HCM':1, 'MINF':2, 'NOR':3, 'RV':4})
df.to_csv('/data/aiiih/projects/nakashm2/css_saliency/data/acdc.csv', index=False)
