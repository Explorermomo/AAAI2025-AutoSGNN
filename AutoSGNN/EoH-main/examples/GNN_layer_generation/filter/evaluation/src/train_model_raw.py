#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

import os
ABS_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(ABS_PATH, "../../.."))
Generation_Path = os.path.join(ROOT_PATH,  'result/LLM_generation/generate_pop/generate_pop.json')
# print(Generation_Path)

import argparse
from dataset_utils import *
from utils import random_planetoid_splits, random_graph_splits
from GNN_models import *
from torch_geometric.loader import DataLoader as DATALOADER

import torch
import torch.nn.functional as F
from tqdm import tqdm

import sys
import json
import types
import warnings
import numpy as np
import random
import time


import torch.utils.data
import time

import csv

import random

from sklearn.model_selection import StratifiedKFold, train_test_split


def get_all_split_idx(dataset):
    """
        - Split total number of graphs into 3 (train, val and test) in 80:10:10
        - Stratified split proportionate to original distribution of data with respect to classes
        - Using sklearn to perform the split and then save the indexes
        - Preparing 10 such combinations of indexes split to be used in Graph NNs
        - As with KFold, each of the 10 fold have unique test set.
    """
    root_idx_dir = './data/TUs/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}

    # If there are no idx files, do the split and store the files
    if not (os.path.exists(os.path.join(root_idx_dir, dataset.name + '_train.index'))):
        print("[!] Splitting the data into train/val/test ...")

        # Using 10-fold cross val to compare with benchmark papers
        k_splits = 10

        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        k_data_splits = []

        graph_labels = [data.y.item() for data in dataset]

        for indexes in cross_val_fold.split(range(len(dataset)), graph_labels):
            remain_index, test_index = indexes[0], indexes[1]

            remain_set = [dataset[i] for i in remain_index]
            remain_labels = [graph_labels[i] for i in remain_index]

            # Gets final 'train' and 'val'
            train_idx, val_idx = train_test_split(remain_index,
                                                  test_size=0.111,
                                                  stratify=remain_labels)
            
            idx_train = list(train_idx)
            idx_val = list(val_idx)
            idx_test = list(test_index)

            with open(os.path.join(root_idx_dir, dataset.name + '_train.index'), 'a+') as f_train, \
                 open(os.path.join(root_idx_dir, dataset.name + '_val.index'), 'a+') as f_val, \
                 open(os.path.join(root_idx_dir, dataset.name + '_test.index'), 'a+') as f_test:
                
                csv.writer(f_train).writerow(idx_train)
                csv.writer(f_val).writerow(idx_val)
                csv.writer(f_test).writerow(idx_test)

        print("[!] Splitting done!")

    # reading idx from the files
    for section in ['train', 'val', 'test']:
        with open(os.path.join(root_idx_dir, dataset.name + '_' + section + '.index'), 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


def RunExp_GC(args, dataset, data, Net, percls_trn, val_lb, Layer_module,train_loader,val_loader,test_loader):

    def train(model, optimizer, loader, device):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def test(model, loader, device):
        model.eval()
        accs, losses = [], []
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.max(1)[1]

            acc = pred.eq(data.y).sum().item() / data.num_graphs
            loss = F.nll_loss(out, data.y)

            accs.append(acc)
            losses.append(loss.item())
        return sum(accs) / len(accs), sum(losses) / len(losses)
    

    appnp_net = Net(dataset.num_features, dataset.num_classes, Layer_module)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = appnp_net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 10000
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        # print(epoch)
        train_loss = train(model, optimizer, train_loader, device)
        val_acc, val_loss = test(model, val_loader, device)
        tmp_test_acc, _ = test(model, test_loader, device)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        if args.early_stopping > 0 and epoch > args.early_stopping:
            tmp = torch.tensor(val_loss_history[-(args.early_stopping + 1):-1])
            if val_loss > tmp.mean().item():
                break

    return test_acc, best_val_acc, best_val_loss, train_loss

def RunExp(args, dataset, data, Net, percls_trn, val_lb, Layer_module):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data.x,data.edge_index)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data.x,data.edge_index), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]

            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(model(data.x,data.edge_index)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset.num_features, dataset.num_classes, Layer_module)
    device = torch.device('cuda')

    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break

    return test_acc, best_val_acc,best_val_loss,train_loss

def reading_code(load_pop_path,index=0):
    try:
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conditate_population = []
            with open(load_pop_path) as file:
                data = json.load(file)
            for individual in data:
                conditate_population.append(individual)

            Model_layer_code = conditate_population[index]

            # Create a new module object
            Layer_module = types.ModuleType("Layer_module")
            exec(Model_layer_code["code"], Layer_module.__dict__)
            # Add the module to sys.modules so it can be imported
            sys.modules[Layer_module.__name__] = Layer_module

            return Layer_module.GNN_Layer
    except Exception as e:
        print("Error-------------------111:", str(e))
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01) # 0.01 0.05
    parser.add_argument('--weight_decay', type=float, default=0.001) #  0.001 0.005 0.01
    parser.add_argument('--early_stopping', type=int, default=200) 
    parser.add_argument('--hidden', type=int, default=64)   # NC 64   # GC 32  16
    parser.add_argument('--dropout', type=float, default=0.5)  # 0 0.1 0.2 0.5 0.7 0.8 0.9
    parser.add_argument('--train_rate', type=float, default=0.6)  # or the dense splitting ratio (60%/20%/20%) for the full-supervised learning setting.
    parser.add_argument('--val_rate', type=float, default=0.2) 
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--Task', default='NC')  # NC, GC
    parser.add_argument('--dataset', default='PubMed') 
    parser.add_argument('--batch_size', default=300) 

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str ,default='LLM_FILTER')
    args = parser.parse_args()

    JSON_file = Generation_Path
    # JSON_file = '/home/jing/MO/the_5th_paper_AutoSGNN/filter/GPRGNN-master/src/LLM_generation/conditate_population/conditate_pops.json'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    start_time = time.time()


    Net = LLM_FILTER


    if args.Task == 'NC':

        dname = args.dataset
        dataset, data = DataLoader(dname)

        RPMAX = args.RPMAX
        Init = args.Init

        alpha = args.alpha
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
        print('True Label rate: ', TrueLBrate)

        args.C = len(data.y.unique())

        Results0 = []

        Layer_module = reading_code(JSON_file, index=args.index)

        for RP in tqdm(range(RPMAX)):

            test_acc, best_val_acc,best_val_loss,train_loss  = RunExp(
                args, dataset, data, Net, percls_trn, val_lb, Layer_module)
            Results0.append([test_acc, best_val_acc])
        endT = time.time()
    
    elif args.Task == 'GC':
        dname = args.dataset
        dataset, data = DataLoader_GC(dname)

        RPMAX = args.RPMAX
        Init = args.Init
        alpha = args.alpha
        train_rate = args.train_rate
        val_rate = args.val_rate
        percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
        val_lb = int(round(val_rate*len(data.y)))
        args.C = len(data.y.unique())
        Results0 = []

        Layer_module = reading_code(JSON_file, index=args.index)

        all_idx = get_all_split_idx(dataset)

        for RP in tqdm(range(RPMAX)):

            train_loader = DATALOADER([dataset[i] for i in all_idx['train'][RP]], batch_size=args.batch_size,  shuffle=True)
            val_loader = DATALOADER([dataset[i] for i in all_idx['val'][RP]], batch_size=args.batch_size, shuffle=False)
            test_loader = DATALOADER([dataset[i] for i in all_idx['test'][RP]], batch_size=args.batch_size, shuffle=False)

            test_acc, best_val_acc,best_val_loss,train_loss  = RunExp_GC(
                args, dataset, data, Net, percls_trn, val_lb, Layer_module,train_loader,val_loader,test_loader)
            Results0.append([test_acc, best_val_acc])
        endT = time.time()


    test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
    test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100
    print(f"++++++++++++++++++++++++{args.index}++++++++++++++++++++++++++++++++++++++")
    print(f'AutoSGNN on dataset {args.dataset}, in {RPMAX} repeated experiment:')
    print(
        f'test acc mean = {test_acc_mean:.4f} \t test acc std = {test_acc_std:.4f} \t val acc mean = {val_acc_mean:.4f} \t val loss = {best_val_loss:.4f} \t train loss = {train_loss:.4f}')
    print(f"==========================================================================")

    try:
        with open(JSON_file, 'r') as json_file:
            data = json.load(json_file)
        data[args.index]["objective"] = np.round(val_acc_mean, 5)
        with open(JSON_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        # startT_file =  '/home/jing/MO/the_5th_paper_AutoSGNN_paper_version/EoH-main/examples/GNN_layer_generation/filter/GPRGNN-master/src/some_time_record/startT.json'
        # with open(startT_file) as file:
        #     startT = json.load(file)
        # Each_popT = '/home/jing/MO/the_5th_paper_AutoSGNN_paper_version/EoH-main/examples/GNN_layer_generation/filter/GPRGNN-master/src/some_time_record/each_pops.json'
        # with open(Each_popT) as file:
        #     each_popT = json.load(file)
        # each_popT['Time'].append(endT-startT)
        # each_popT['Value'].append(np.round(test_acc_mean, 5))
        # with open(Each_popT, 'w') as f:
            # json.dump(each_popT, f, indent=5)
    except Exception as e:
        #print("Error:", str(e))
        print("Error-------------------333:", str(e))
