import copy
import math

import numpy as np
import pydicom
from scipy import ndimage
from scipy.ndimage import zoom
from skimage.measure import regionprops
from skimage import exposure

def resize_4darray(image, segment, crop_factor, nDimLimit, center, ord):
    """
    Align slice and zoom up

    Inputs
        image: image to be operated on(npy)
        segment: autosegmentation array(npy)
        nTimeLimit: time limits of the img
    Outputs
        imgOut: resized video
    """
    
    nRows, nCols, nSlices, nFrames = image.shape
    ### crops or pads slice data
    if nSlices >= nDimLimit[2]:
        image = image[:,:,0:nDimLimit[2],:]
    else:
        nPadBefore = math.floor((nDimLimit[2]-nSlices)/2)
        nPadAfter = nDimLimit[2]-nSlices - nPadBefore
        image = np.pad(image,((0,0),(0,0),(nPadBefore,nPadAfter),(0,0)),'constant')
    ### crops or pads frame data
    if nFrames >= nDimLimit[3]:
        image = image[:,:,:,0:nDimLimit[3]]
    else:
        ### adding black images after
        nPad = math.floor(nDimLimit[3]-nFrames)
        image = np.pad(image,((0,0),(0,0),(0,0),(0,nPad)),'constant')

    if center:
        ### Get center of segmentation and crop image
        segment_cp = copy.deepcopy(segment)
        x = segment_cp.reshape((nRows, nCols, segment_cp.shape[2]*segment_cp.shape[3]))
        x[x != 2] = 0
        x[x == 2] = 1        
        properties = regionprops(x.astype(int))
        cen = properties[0].centroid

        nRowsLen = np.round(nRows/crop_factor*0.5)
        nColsLen = np.round(nCols/crop_factor*0.5)
        
        nRows1 = int(cen[0]-nRowsLen)
        nCols1 = int(cen[1]-nColsLen)
        nRows2 = int(cen[0]+nRowsLen)
        nCols2 = int(cen[1]+nColsLen)

        ### If index is out of range, zero padding to the image
        if nRows1 < 0:
            image = np.pad(image,((abs(nRows1),0),(0,0),(0,0),(0,0)),'constant')
            nRows1 = 0
        if nCols1 < 0:
            image = np.pad(image,((0,0),(abs(nCols1),0),(0,0),(0,0)),'constant')
            nCols1 = 0
        if nRows2 > nRows:
            image = np.pad(image,((0,abs(nRows2)),(0,0),(0,0),(0,0)),'constant')
        if nCols2 > nRows:
            image = np.pad(image,((0,0),(0,abs(nCols2)),(0,0),(0,0)),'constant')
            
        image = image[nRows1:nRows2, nCols1:nCols2, :, :]

    elif not center and crop_factor!=1:
        ### crop image to center middle
        nRowsCrop = np.round(nRows/crop_factor)
        nColsCrop = np.round(nCols/crop_factor)
        nRows1 = int(nRows - nRowsCrop)//2
        nCols1 = int(nCols - nColsCrop)//2
        nRows2 = int(nRows1 + nRowsCrop)
        nCols2 = int(nCols1 + nColsCrop)
        image = image[nRows1:nRows2, nCols1:nCols2, :, :]
    else:
        pass
    nRows, nCols, nSlices, nFrames = image.shape

    ### Make empty images all zero
    np_empty =np.reshape(image, (image.shape[0]*image.shape[1], nSlices, nFrames)).sum(axis=0)
    bool_arr = (np_empty == 0)

    if (nRows!=nDimLimit[0]) or (nCols!=nDimLimit[1]):
        factorZoom = (nDimLimit[0]/nRows,nDimLimit[1]/nCols,1,1)
        imgOut = zoom(image, factorZoom, order=ord)
    else:
        imgOut=image

    ### Make sure empty images are all black pixels
    imgOut[:,:,bool_arr] = 0

    return imgOut



def padding(image, nRows, nCols):
    """
    pads image to square to not fuck up the zoom
    Inputs
        image: image to be operated on(npy)
        nRows: X size of image(int) 
        nCols: Y size of image(int) 
    Outpit
        image: image to be operated on(npy)
        nRows: X size of image(int)  
        nCols: Y size of image(int)  
    """
    if (nRows/nCols)<1:
        newRows = int(nCols)
        nPadBefore = int(math.floor((newRows-nRows)/2))
        nPadAfter = newRows - nRows - nPadBefore
        image = np.pad(image,((nPadBefore,nPadAfter),(0,0)),'constant')
        nRows = newRows

    if (nRows/nCols)>1:
        newCols = int(nRows)
        nPadBefore = int(math.floor((newCols-nCols)/2))
        nPadAfter = newCols - nCols - nPadBefore
        image = np.pad(image,((0,0),(nPadBefore,nPadAfter)),'constant')
        nCols = newCols
    return image, nRows, nCols


def resize_image(image, new_pixel_space, zoom_factor, nDimLimit, ord):
    """
    Zoom in and out based on cropfactor and resize images
    Inputs
        image: image to be operated on(npy)
        nDimLimit: limits of the img. Expects it to be Rows x Columns(tuple)
        zoom_factor:  tuple
        ord: zoom order

    Outputs
        imgOut: resized image
    """

    nRows_original, nCols_original = image.shape        

    image = zoom(image, zoom_factor, order=ord) 
    image, nRows, nCols = padding(image, *image.shape)

    if new_pixel_space is None:
        crop_factor=1
    else:
        if (nRows_original/nCols_original)<1:
            crop_factor = nCols / nDimLimit[0]
        else:
            crop_factor = nRows / nDimLimit[0]


    if crop_factor > 1:
        nRowsCrop = np.round(nRows/crop_factor)
        nColsCrop = np.round(nCols/crop_factor)
        nRows1 = int(nRows - nRowsCrop)//2
        nCols1 = int(nCols - nColsCrop)//2
        nRows2 = int(nRows1 + nRowsCrop)
        nCols2 = int(nCols1 + nColsCrop)
        image = image[nRows1:nRows2, nCols1:nCols2]

    ### zoom out
    elif crop_factor< 1:
        nRows1 = int(nDimLimit[0] - nRows)//2
        nCols1 = int(nDimLimit[1] - nCols)//2
        nRows2 = int(nRows + nRows1)
        nCols2 = int(nCols + nCols1)
        # Zero-padding
        out = np.zeros(nDimLimit)
        out[nRows1:nRows2, nCols1:nCols2] = image
        image = out

    else:
        pass
    
    nRows, nCols = image.shape

    ## resizes image to the input limits
    if (nRows!=nDimLimit[0]) or (nCols!=nDimLimit[1]):
        print(nRows, nCols)
        factorZoom = (nDimLimit[0]/nRows,nDimLimit[1]/nCols)
        imgOut = zoom(image, factorZoom, order=ord)
        
    else:
        imgOut=image
    return imgOut


def get_center(df_pt):
    cen_lst = []
    for phase in np.sort(df_pt['PhaseNumber'].unique()):
        for slice in np.sort(df_pt['SliceLocation'].unique()):
            df_pt_1 = df_pt[(df_pt['PhaseNumber']==phase) & (df_pt['SliceLocation']==slice)]
            if len(df_pt_1)==0:
                continue
            seg_img = np.load(df_pt_1.iloc[0]['Seg_path'])
            seg_img[seg_img != 2] = 0
            properties = regionprops(seg_img.astype(int))
            try:
                cen_lst.append(properties[0].centroid)
            except IndexError as error:
                continue
    cen_lst = np.array(cen_lst)

    if len(cen_lst) ==0:
        return None
    else:    
        return np.mean(cen_lst, axis=0)

def move_center_heart(image, center):
    ### Center
    nRows, nCols = image.shape
    nRowsLen = int(nRows//2)
    nColsLen = int(nCols//2)
    
    nRows1 = int(center[0]-nRowsLen)
    nCols1 = int(center[1]-nColsLen)
    nRows2 = int(center[0]+nRowsLen)
    nCols2 = int(center[1]+nColsLen)

    # print(nRows1,nRows2, nCols1,nCols2)
    ### If index is out of range, zero padding to the image
    if nRows1 < 0:
        image = np.pad(image,((abs(nRows1),0),(0,0)),'constant')
        nRows2 += abs(nRows1) +1
        nRows1 = 0
    if nCols1 < 0:
        image = np.pad(image,((0,0),(abs(nCols1),0)),'constant')
        nCols2 += abs(nCols1) +1
        nCols1 = 0
    if nRows2 > nRows:
        image = np.pad(image,((0,abs(nRows2-nRows)),(0,0)),'constant')
    if nCols2 > nCols:
        image = np.pad(image,((0,0),(0,abs(nCols2-nCols))),'constant')
    
    image = image[nRows1:nRows2, nCols1:nCols2]
    return image


def read_n_process_img(nX, nY, df_pt_1, new_pixel_space, center=None, segment=True):
    """
    read MRI and segmentations, and then remove outlier, normalize, resize images
    Inputs
        nX:: X size of image (int)
        nY: Y size of image (int)
        df_pt_1: patient info (dataframe)
    Outputs
        np_img: MR image (numpy)
        seg_img: segmentation image (numpy)
    """
    HE= False

    if segment:
        try:
            np_img = np.load(df_pt_1['Image_path'])
            seg_img = np.load(df_pt_1['Seg_path'])
        except:
            return None, None

        if np_img.sum()==0:
            return None, None

        if HE:
            np_img = exposure.equalize_adapthist(np_img, kernel_size=None, clip_limit=0.01)
        # else:
        #     min_value, max_value = [np.mean(np_img) - 3 * np.std(np_img), np.mean(np_img) + 3 * np.std(np_img)]
        #     np_img[np_img<min_value] = min_value
        #     np_img[np_img>max_value] = max_value

        ### move heart to the center
        if center is not None:
            np_img = move_center_heart(np_img, center)
            seg_img = move_center_heart(seg_img, center)

        zoom_factor = (df_pt_1['zoom_factor_X'],df_pt_1['zoom_factor_Y'])

        ### Normalize the resolution
        np_img = resize_image(
            np_img, new_pixel_space, zoom_factor=zoom_factor, 
            nDimLimit=(nX,nY), ord=3
        )
        seg_img = resize_image(
            seg_img, new_pixel_space, zoom_factor=zoom_factor, 
            nDimLimit=(nX,nY), ord=0
        )
        return np_img, seg_img
    
    else: 
        try:
            np_img = pydicom.dcmread(df_pt_1['DCM_path']).pixel_array
        except:
            return None

        if np_img.sum()==0:
            return None

        if HE:
            np_img = exposure.equalize_adapthist(np_img, kernel_size=None, clip_limit=0.01)
        # else:
        #     ## Remove outlier
        #     min_value, max_value = [np.mean(np_img) - 3 * np.std(np_img), np.mean(np_img) + 3 * np.std(np_img)]
        #     np_img[np_img<min_value] = min_value
        #     np_img[np_img>max_value] = max_value

        zoom_factor = df_pt_1['zoom_factor_X']

        ### Normalize the resolution
        np_img = resize_image(
            np_img, new_pixel_space, zoom_factor=zoom_factor, 
            nDimLimit=(nX,nY), ord=3
        )
    return np_img


def norm(img):
    """Normalize images"""
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def check_npy(npy_array, accession_num, df, view, MIN_PHASE, MIN_SLICE):
    """
    Check if MRI has enough phases
    Input
        npy_array: 4D MRI array(np.array)
        min_phase: Minimum phase requirement (int)
        accession_num: accession number(int)
        df: Dataframe(pandas)
        MIN_PHASE: int
        MIN_SLICE: int
    Output:
        Return Bool
    """

    if len(df[df['AccessionNumber']==accession_num])==0:
        return False
        
    ### Check how many middle phase images are available
    numPhases = npy_array.shape[3]
    numSlice = npy_array.shape[2]
    if view in ['cine_sa']:
        ed_phase_header = 'Clinical Results LV PhaseDiastole'
        es_phase_header = 'Clinical Results LV PhaseSystole'
        ### Check how ED and ES images are available
        ed_img = npy_array[:,:,:,int(df[ed_phase_header][df['AccessionNumber']==accession_num].values[0])-1]
        es_img = npy_array[:,:,:,int(df[es_phase_header][df['AccessionNumber']==accession_num].values[0])-1]
        ed_sum = np.sum(ed_img, (0,1)) 
        es_sum = np.sum(es_img, (0,1)) 
        numSliceED = np.count_nonzero(ed_sum)
        numSliceES = np.count_nonzero(es_sum)
        if (numPhases < MIN_PHASE) | (numSliceED < MIN_SLICE)| (numSliceES < MIN_SLICE):
            return False
        else:
            return True
    elif view in 'cine_2ch':
        ed_phase_header = 'Monoplanar 2CV *** Phase LV Diastole'
        es_phase_header = 'Monoplanar 2CV *** Phase LV Systole'
        if (numPhases < MIN_PHASE):
            return False
        else:
            return True

    elif view in 'cine_3ch':
        ed_phase_header = 'Clinical Results LV PhaseDiastole'
        es_phase_header = 'Clinical Results LV PhaseSystole'
        if (numPhases < MIN_PHASE):
            return False
        else:
            return True

    elif view in 'cine_lax':
        ed_phase_header = 'Monoplanar 4CV *** Phase LV Diastole'
        es_phase_header = 'Monoplanar 4CV *** Phase LV Systole'
        if (numPhases < MIN_PHASE):
            return False
        else:
            return True
        
    elif view in ['lge_sa']:
        if (numSlice < MIN_SLICE):
            return False
        else:
            return True

def add_channel(img):
    """ 
    Make 0 channel images to 3 channels images
    Input 
        img: gray scale numpy array (np.array)
    Outpit
        img_channel: 3 channels numpy arrat (np.array)
    """
    if len(img.shape) ==2:
        img_channel = np.empty((*img.shape, 3))
        img_channel[:,:,0] = img
        img_channel[:,:,1] = img
        img_channel[:,:,2] = img
  
    elif len(img.shape) ==3:
        img_channel = np.empty((*img.shape, 3))
        img_channel[:,:,:,0] = img
        img_channel[:,:,:,1] = img
        img_channel[:,:,:,2] = img
        
        img_channel =  np.transpose(img_channel, axes=[2, 0, 1, 3])
    return img_channel


def resampling_4darray(img ,nDimLimit, ord):
    desired_phase = nDimLimit[3]
    desired_slice = nDimLimit[2]
    desired_height = nDimLimit[1]
    desired_width = nDimLimit[0]

    # Get current depth
    current_phase = img.shape[3]
    current_slice = img.shape[2]
    current_height = img.shape[1]
    current_width = img.shape[0]

    # Compute depth factor
    height = current_height / desired_height
    width = current_width / desired_width

    height_factor = 1 / height
    width_factor = 1 / width

    img_lst = []
    for i in range(img.shape[2]):
        
        ### remove empty array
        idx_lst = [idx for idx, s in enumerate(np.sum(img[:,:,i,:], (0,1))) if s != 0]
        if len(idx_lst) < desired_slice//2:
            continue
        phase = len(idx_lst) / desired_phase
        phase_factor = 1 / phase
        img_slice = ndimage.zoom(img[:, :, i , idx_lst], (width_factor, height_factor, phase_factor), order=ord)
        img_lst.append(img_slice)

    ndarrays = np.array(img_lst)
    ndarrays =  np.transpose(ndarrays, axes=[1, 2, 0, 3])
    slice = ndarrays.shape[2] / desired_slice
    slice_factor = 1 / slice

    img = ndimage.zoom(ndarrays, (width_factor, height_factor, slice_factor, 1), order=ord)
    return img
