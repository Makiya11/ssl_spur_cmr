import sys
import glob
import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.append('/data/aiiih/projects/nakashm2/css_saliency/code_publish')
from utils.func import roc_auc_ci

def get_auc(df):
    diagnosis_num = df['diagnosis_num']   
    if 'ccf_cm' in data_name:
        classes = ['probs_icm','probs_nicm','probs_amyl','probs_hcm'] 
        y_prob = df[classes]
    elif 'acdc' in data_name:
        classes = ['probs_DCM','probs_HCM','probs_MINF','probs_NOR','probs_RV']
        y_prob = df[classes]
        
    AUC_sum, std_sum = 0, 0   
    for idx, i_class in enumerate(classes):
        y_true = (diagnosis_num==idx)*1
        y_score = y_prob[i_class]
        try:
            AUC, std = roc_auc_ci(y_true, y_score, positive=1)
        except: 
            # handle the exception
            continue
        AUC_sum += AUC
        std_sum += std    
    AUC = AUC_sum/len(classes)
    std = std_sum/len(classes) 
    num_pt = len(y_prob)
    return AUC, std, num_pt

root_dir = '/data/aiiih/projects/nakashm2/css_saliency'
model_name = 'DenseNet121'
for phase in ['linear', 'finetune']:
    for data_name in ['acdc', 'ccf_cm']:
        dice_lst = []
        for csv_name in glob.glob(f'{root_dir}/output2/{data_name}/{phase}/gradcam/*/{model_name}.csv'):
            df = pd.read_csv(csv_name)
            df = df[df['dice']!=0]
            df = df.dropna(subset='dice')

            # df['dice'] = df['dice'].fillna(0)
            # df['ark'] = df['ark'].fillna(0)
            dice_ci = st.t.interval(alpha=0.95, df=len(df['dice'])-1, loc=np.mean(df['dice']), scale=st.sem(df['dice'])) 
            ark_ci = st.t.interval(alpha=0.95, df=len(df['ark'])-1, loc=np.mean(df['ark']), scale=st.sem(df['ark'])) 
            auc, std, num = get_auc(df)
            dic = {
                'experiment_name': csv_name.split('/')[-2], 
                'phase': df['phase'].iloc[0],  
                'auc':auc, 
                'std':std, 
                'num':num, 
                'dice': f"{df['dice'].mean().round(3)}±{(dice_ci[1]-df['dice'].mean()).round(3)}",
                'ark': f"{df['ark'].mean().round(3)}±{(dice_ci[1]-df['ark'].mean()).round(3)}",
            }
            dice_lst.append(dic)
        dice_df= pd.DataFrame(dice_lst).round(decimals=3)
        breakpoint()


