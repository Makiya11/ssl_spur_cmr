import sys
from math import sqrt
from sklearn.metrics import roc_auc_score

def get_input_datasize(model_name):
    model_transform_params  = {
            "DenseNet121": {
                "side_size": 224,
                "num_frames":1
            },
            
            "VGG16": {
                "side_size": 224,
                "num_frames":1
            }
        }
    transform_params = model_transform_params[model_name]
    return transform_params

def str_flag(value):
    if value.upper() == 'NONE':
        return None
    elif value.upper() == 'TRUE':
        return True
    elif value.upper() == 'FALSE':
        return False
    return value


def duplicate_check(df, params):
    df = df.fillna('None')
    df2 = df[(df['dl_model'] == params['dl_model']) &
                (df['opt']== params['opt']) &
                (df['batch_size']== params['batch_size']) &
                (df['max_epochs']== params['max_epochs']) &
                (df['dataset_name']== params['dataset_name']) &
                (df['classification']== params['classification']) &
                (df['learning_rate']== params['learning_rate']) &
                (df['dataset_name']== params['dataset_name']) & 
                (df['pt_dataset_name']== ('None' if params['pt_dataset_name'] is None else params['pt_dataset_name'])) &
                (df['pt_weight']== ('None' if params['pt_weight'] is None else params['pt_weight'])) &
                (df['pt_init_weight']== ('None' if params['pt_init_weight'] is None else params['pt_init_weight'])) &
                (df['pt_batch_size']== ('None' if params['pt_batch_size'] is None else params['pt_batch_size'])) &
                (df['pt_learning_rate']== ('None' if params['pt_learning_rate'] is None else params['pt_learning_rate'])) &
                (df['input_img']== ('None' if params['input_img'] is None else params['input_img'])) &
                (df['pt_input_img']== ('None' if params['pt_input_img'] is None else params['pt_input_img'])) &
                (df['feature_extract']== ('None' if params['feature_extract'] is None else params['feature_extract']))]    
    if len(df2) !=0:
        print(f'Already done')
        sys.exit()

def get_runname(params):
    name = f"{params['dl_model']}_" \
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
    return name
             
def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score, average='micro')
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return AUC, AUC-lower