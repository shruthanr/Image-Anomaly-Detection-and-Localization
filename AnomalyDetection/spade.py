import torch
import torch.nn.functional as F

from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def spade_localization(train_outputs, test_outputs, topk_indexes, class_name):
    score_map_list = []
    for t_idx in tqdm(range(test_outputs['avgpool'].shape[0]), '| localization | test | %s |' % class_name):
        score_maps = []
        for layer_name in ['layer1', 'layer2', 'layer3']:  # for each layer

            # construct a gallery of features at all pixel locations of the K nearest neighbors
            topk_feat_map = train_outputs[layer_name][topk_indexes[t_idx]]
            test_feat_map = test_outputs[layer_name][t_idx:t_idx + 1]
            feat_gallery = topk_feat_map.transpose(3, 1).flatten(0, 2).unsqueeze(-1).unsqueeze(-1)

            # calculate distance matrix
            feat_gallery = feat_gallery.transpose(1,2).transpose(2,3)
            test_feat_map = test_feat_map.transpose(1,2).transpose(2,3)
            dist_matrix_list = []
            for d_idx in range(feat_gallery.shape[0] // 100):
                dist_matrix = torch.pairwise_distance(feat_gallery[d_idx * 100:d_idx * 100 + 100], test_feat_map)
                dist_matrix_list.append(dist_matrix)
            dist_matrix = torch.cat(dist_matrix_list, 0)

            # k nearest features from the gallery (k=1)
            score_map = torch.min(dist_matrix, dim=0)[0]
            score_map = F.interpolate(score_map.unsqueeze(0).unsqueeze(0), size=224,
                                        mode='bilinear', align_corners=False)
            score_maps.append(score_map)

        # average distance between the features
        score_map = torch.mean(torch.cat(score_maps, 0), dim=0)

        # apply gaussian smoothing on the score map
        score_map = gaussian_filter(score_map.squeeze().cpu().detach().numpy(), sigma=4)
        score_map_list.append(score_map)
        
    return score_map_list