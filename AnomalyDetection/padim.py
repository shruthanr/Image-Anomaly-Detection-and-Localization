import torch
import numpy as np
from .utils import concatenate_embeddings
import torch.nn.functional as F
import random
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter


def get_embedding_vectors(features, device):
    embeddings = features['layer1']
    for layer in ['layer2', 'layer3']:
        embeddings = concatenate_embeddings(embeddings, features[layer], device)
    return embeddings
    
def gaussian_train(train_features, idx, device):
    
    embedding_vectors = torch.index_select(get_embedding_vectors(train_features, device), 1, idx)
 
    N, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(N, C, H * W)
    mean, cov = torch.mean(embedding_vectors, dim=0).numpy(), torch.zeros(C, C, H*W).numpy()
    for i in range(H*W):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * np.identity(C)
    train_features = [mean, cov]
    return train_features

def gaussian_test(train_features, test_features, idx, image_dims, device):
    
    embedding_vectors = torch.index_select(get_embedding_vectors(test_features, device), 1, idx)
    
    N, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(N, C, H * W).numpy()
    
    dist_list = []
    for i in range(H * W):
        mean, conv_inv = train_features[0][:, i], np.linalg.inv(train_features[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(N, H, W)
    
    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=image_dims[2], mode='bilinear',
                              align_corners=False)
    score_map = score_map.squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)

    # Normalization
    min_score, max_score = score_map.min(), score_map.max()
    scores = (score_map - min_score) / (max_score - min_score)
    
    return scores