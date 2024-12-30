#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask 


def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def random_graph_splits(dataset, num_classes, percls_trn=0.8, val_lb=0.5, Flag=0):

    # percls_trn = int(len(dataset)*0.6)
    # val_lb = int(len(dataset)*0.2)

    indices = [[] for _ in range(num_classes)]
    for idx, data in enumerate(dataset):
        indices[data.y.item()].append(idx)

    train_indices = []
    for i in range(num_classes):
        train_indices += indices[i][:int(len(indices[i])*percls_trn)]

    train_indices = torch.tensor(train_indices)

    rest_indices = []
    for i in range(num_classes):
        rest_indices += indices[i][int(len(indices[i])*percls_trn):]
    rest_indices = torch.tensor(rest_indices)
    rest_indices = rest_indices[torch.randperm(rest_indices.size(0))]

    val_indices = rest_indices[:int(len(rest_indices)*val_lb)]
    test_indices = rest_indices[int(len(rest_indices)*val_lb):]


    return train_indices, val_indices, test_indices