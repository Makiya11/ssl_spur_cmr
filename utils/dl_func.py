import os
import torch
import copy

def fcnSaveCheckpoint(model, optimizer, epoch, metric_value, filepath, params):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    ### saves checkpoint to allow resumption of training
    state = {'epoch': epoch,
             'state_dict': model.state_dict(),
             'best_metric': metric_value,
             'optimizer' : optimizer.state_dict()}
    state.update(params)
    torch.save(state,filepath)

def fcnLoadCheckpoint(model, optimizer, filepath):
    ### loads checkpoint to allow resumption of training
    epoch = 0
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Previous metric: {:.3f}".format(checkpoint['best_metric']))
    else:
        print("Error no checkpoint found: {}".format(filepath))

    return model, optimizer, epoch, checkpoint['best_metric']


def rename_weights(state_dict, idx):
    state_dict_v2 = copy.deepcopy(state_dict)

    rename_key = ['conv0.', 'encoder.']
    
    for key in state_dict:
        
        if (True for i in rename_key if i in key):
            new_name = key.replace(rename_key[0], f'conv0_{idx}.').replace(rename_key[1], f'encoder_{idx}.')
            state_dict_v2[new_name] = state_dict_v2.pop(key)
    return state_dict_v2


def load_partial_weights(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(f'Number of layers in the model {len(model.state_dict())}')

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
    print(f'Number of layers to load {len(pretrained_dict)}')
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model 

class EarlyStopping():
    def __init__(self, patience=10, delta=0, min_epoch=15):        
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.delta = delta
        self.min_epoch = min_epoch
        self.flag_save_model = False
        self.flag_early_stop = False
    
    def update(self, x, epoch):
        if self.best_score is None:
            self.best_score = x
            self.flag_save_model = True
        elif x > self.best_score + self.delta:
            self.counter += 1
            self.flag_save_model = False
            print(f'Early Stopping: {self.counter} / {self.patience}')
            if self.counter >= self.patience and epoch> self.min_epoch:
                self.flag_early_stop = True
        else:
            self.best_score = x
            self.flag_save_model = True
            self.counter = 0
        return self.flag_save_model, self.flag_early_stop
