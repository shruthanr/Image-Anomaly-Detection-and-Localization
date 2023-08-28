import os
import torch
import pickle
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel

    
def get_train_features(train_dataloader, class_name, train_feature_filepath, device='cuda'):

    train_outputs = OrderedDict([('avgpool', [])])
    
    model_name = 'google/vit-base-patch16-224-in21k'
    model = ViTModel.from_pretrained(model_name)
    model.to(device)
    

    for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
        # model prediction
        with torch.no_grad():
            pred = model(x.to(device))
        train_outputs['avgpool'].append(pred['pooler_output'])

    train_outputs['avgpool'] = torch.cat(train_outputs['avgpool'], 0)
    return train_outputs
    

def get_test_features(test_dataloader, class_name, device='cpu'):

    
    gt_list = []
    gt_mask_list = []
    test_imgs = [] 

    test_outputs = OrderedDict([('avgpool', [])])
    
    model_name = 'google/vit-base-patch16-224-in21k'
    model = ViTModel.from_pretrained(model_name)
    model.to(device)
    
    outputs = []

    for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with torch.no_grad():
            pred = model(x.to(device))
        test_outputs['avgpool'].append(pred['pooler_output'])

    test_outputs['avgpool'] = torch.cat(test_outputs['avgpool'], 0)
    
    return test_outputs, gt_list, gt_mask_list, test_imgs

