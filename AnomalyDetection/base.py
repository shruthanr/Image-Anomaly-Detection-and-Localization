import os
import torch
import pickle
from collections import OrderedDict
from tqdm import tqdm
import torch.nn.functional as F

    
def get_train_features(model, train_dataloader, class_name, device='cpu'):
    
    train_features = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    
    outputs = []
    model.layer1[-1].register_forward_hook(lambda mod, inp, out : outputs.append(out))
    model.layer2[-1].register_forward_hook(lambda mod, inp, out : outputs.append(out))
    model.layer3[-1].register_forward_hook(lambda mod, inp, out : outputs.append(out))
    model.avgpool.register_forward_hook(lambda mod, inp, out : outputs.append(out))
    

    for (x, y, mask) in tqdm(train_dataloader, f'| feature extraction | train | {class_name} |'):
        with torch.no_grad():
            pred = model(x.to(device))
        for layer, feat in zip(train_features.keys(), outputs):
            train_features[layer].append(feat)
        outputs = []
    for layer, feat in train_features.items():
        train_features[layer] = torch.cat(feat, 0)

    return train_features
    

def get_test_features(model, test_dataloader, class_name, device='cpu'):
    test_features = OrderedDict([('layer1', []), ('layer2', []), ('layer3', []), ('avgpool', [])])
    
    ground_truth = []
    ground_truth_masks = []
    test_imgs = []
    
    
    outputs = []
    model.layer1[-1].register_forward_hook(lambda mod, inp, out : outputs.append(out))
    model.layer2[-1].register_forward_hook(lambda mod, inp, out : outputs.append(out))
    model.layer3[-1].register_forward_hook(lambda mod, inp, out : outputs.append(out))
    model.avgpool.register_forward_hook(lambda mod, inp, out : outputs.append(out))

    for (x, y, mask) in tqdm(test_dataloader, f'| feature extraction | test | {class_name} |'):
        test_imgs.extend(x.cpu().detach().numpy())
        ground_truth.extend(y.cpu().detach().numpy())
        ground_truth_masks.extend(mask.cpu().detach().numpy())

        with torch.no_grad():
            pred = model(x.to(device))
   
        for layer, feat in zip(test_features.keys(), outputs):
            test_features[layer].append(feat)
  
        outputs = []

    for layer, feat in test_features.items():
        test_features[layer] = torch.cat(feat, 0)

    
    return test_features, ground_truth, ground_truth_masks, test_imgs

