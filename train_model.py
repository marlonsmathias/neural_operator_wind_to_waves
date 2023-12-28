import torch
import numpy as np
import argparse

import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from utils.dataloader import *
from utils.nop import *
from utils.utils import *
from typing import Dict

import time


# -----------------------------------------
# LOAD ARGUMENTS
def get_args():
    parser = argparse.ArgumentParser(
        prog='Training step',
        usage='%(prog)s [options] parser',
        description='Parser for hyperparams training')
    
    parser.add_argument('--datapath',
                        type=str,
                        default='data/era5_Santos_2022-2023.nc',
                        help='Use to manually select the data file name') 
    
    parser.add_argument('--bathpath',
                        type=str,
                        default='data/era5_Santos_2022-2023_bath.nc',
                        help='Use to manually select the data file name') 
    
    parser.add_argument('--mesh_n_min',
                        type=int,
                        default=250,
                        help='Number of mesh points')

    parser.add_argument('--mesh_n_max',
                        type=int,
                        default=1000,
                        help='Number of mesh points')

    parser.add_argument('--radius',
                        type=float,
                        default=500.e3,
                        help='Maximum edge length in the graph')

    parser.add_argument('--model_width',
                        type=int,
                        default=16,
                        help='')

    parser.add_argument('--model_kernel_width',
                        type=int,
                        default=18,
                        help='')

    parser.add_argument('--model_depth',
                        type=int,
                        default=5,
                        help='')
    
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Maximum number of epochs for training')

    parser.add_argument('--patience',
                        type=int,
                        default=50,
                        help='Number of epochs without improvement in loss for early stopping')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='')

    parser.add_argument('--trainfrac',
                        type=float,
                        default=0.8,
                        help='')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate for the optimization algorithm')

    parser.add_argument('--scheduler_step',
                        type=int,
                        default=100,
                        help='')

    parser.add_argument('--scheduler_gamma',
                        type=float,
                        default=0.9,
                        help='')

    parser.add_argument('--dev',
                        type=str,
                        default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device to run the model')

    parser.add_argument('--seed',
                    type=int,
                    default=None,
                    help='')

    parser.add_argument('--comment',
                    type=str,
                    default='',
                    help='String to be added at the end of the model file name')
    
    args = parser.parse_args()
    
    return args

# -----------------------------------------
# PARAMETERS

args = get_args()
pars = dict()

# Data
pars['data_path'] = args.datapath
pars['bath_path'] = args.bathpath
pars['train_frac'] = args.trainfrac

pars['mesh'] = {
    'n_min': args.mesh_n_min, # Minimum number of sample nodes in the domain
    'n_max': args.mesh_n_max, # Maximum number of sample nodes in the domain
    'radius': args.radius} # Maximum edge length in the graph

# Model
pars['model'] = {
    'width': args.model_width,
    'kernel_width': args.model_kernel_width,
    'depth': args.model_depth}

# Training
pars['train'] = {
    'batch_size': args.batch_size,
    'epochs': args.epochs,
    'patience': args.patience,
    'learning_rate': args.lr,
    'scheduler_step': args.scheduler_step,
    'scheduler_gamma': args.scheduler_gamma}

device = args.dev

# -----------------------------------------
# PRE-PROCESSING

# Set random seed
if args.seed is not None:
    random.seed(args.seed)

# Set up model
model = KernelNN(pars['model']['width'], pars['model']['kernel_width'], pars['model']['depth'], 3, in_width=3, out_width=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=pars['train']['learning_rate'], weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=pars['train']['scheduler_step'], gamma=pars['train']['scheduler_gamma'])

# Load data and generate meshes
d = data_loader(pars['data_path'],pars['bath_path'],pars['train_frac'])
data_train = []
for i in range(d.n_train):

    n = randintlog(pars['mesh']['n_min'],pars['mesh']['n_max'])

    data_train.append(d.sample_graph(n, i, radius=pars['mesh']['radius']))

    print(f'Sample: {i} of {d.n_train}, Nodes: {n}, Edges: {data_train[-1].edge_index.shape[1]}')

loader_train = DataLoader(data_train, batch_size=pars['train']['batch_size'], shuffle=True)


# -----------------------------------------
# TRAINING
ls = []
start_time = time.time()

model.train()

loss_min = 1e10
loss_min_epoch = 0

for epoch in range(pars['train']['epochs']):
    train_mse = 0.
    for batch in loader_train:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        mse.backward()

        optimizer.step()
        train_mse += mse.item()/d.n_train

    print(epoch)
    print(train_mse)

    scheduler.step()

    ls.append(train_mse)

    if (epoch+1)%100==0:
        memory_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
        save_model(model, pars, ls, memory_use, start_time=start_time, seed=args.seed, comment=args.comment)

    if train_mse < loss_min:
        loss_min = train_mse
        loss_min_epoch = epoch
    elif epoch - loss_min_epoch > pars['train']['patience']:
        memory_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
        save_model(model, pars, ls, memory_use, start_time=start_time, seed=args.seed, comment=args.comment)
        break