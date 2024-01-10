import torch
import numpy as np
import argparse
from parfor import parfor

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
                        default=600,
                        help='Number of mesh points')

    parser.add_argument('--mesh_n_max',
                        type=int,
                        default=1250,
                        help='Number of mesh points')

    parser.add_argument('--radius',
                        type=float,
                        default=350.,
                        help='Maximum edge length in the graph')

    parser.add_argument('--model_width',
                        type=int,
                        default=20,
                        help='')

    parser.add_argument('--model_kernel_width',
                        type=int,
                        default=40,
                        help='')

    parser.add_argument('--model_depth',
                        type=int,
                        default=8,
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
                        default=10,
                        help='')

    parser.add_argument('--trainfrac',
                        type=float,
                        default=0.8,
                        help='')
    
    parser.add_argument('--distance_to_sea',
                        type=float,
                        default=300.,
                        help='')

    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate for the optimization algorithm')

    parser.add_argument('--scheduler_step',
                        type=int,
                        default=1e6,
                        help='')
    
    parser.add_argument('--reload_samples',
                        type=int,
                        default=None,
                        help='')

    parser.add_argument('--scheduler_gamma',
                        type=float,
                        default=0.5,
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
    
    parser.add_argument('--n_cases',
                    type=int,
                    default=None,
                    help='Number of total cases used in training - Defaults to all 2552 cases')
    
    args = parser.parse_args()
    
    return args

# -----------------------------------------
# PARAMETERS

if __name__ == '__main__':

    num_workers = 16

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
        'n_cases' : args.n_cases,
        'distance_to_sea' : args.distance_to_sea,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'learning_rate': args.lr,
        'scheduler_step': args.scheduler_step,
        'scheduler_gamma': args.scheduler_gamma,
        'reload_samples': args.reload_samples}

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
    d = data_loader(pars['data_path'],pars['bath_path'],pars['train_frac'], n_cases=pars['train']['n_cases'])

    print('Loading train samples')
    @parfor(range(d.n_train))
    def data_train(i):
        n = randintlog(pars['mesh']['n_min'],pars['mesh']['n_max'])
        return d.sample_graph(n, i, radius=pars['mesh']['radius'])

    print('Loading validation samples')
    @parfor(range(d.n_val))
    def data_val(i):
        n = randintlog(pars['mesh']['n_min'],pars['mesh']['n_max'])
        return d.sample_graph(n, i, radius=pars['mesh']['radius'],validation=True)

    loader_train = DataLoader(data_train, batch_size=pars['train']['batch_size'], shuffle=True)
    loader_val = DataLoader(data_val, batch_size=pars['train']['batch_size'], shuffle=False)


    # -----------------------------------------
    # TRAINING
    ls = []
    ls_val = []
    start_time = time.time()

    model.train()

    loss_min = 1e10
    loss_min_epoch = 0

    for epoch in range(pars['train']['epochs']):
        train_loss = 0.
        val_loss = 0.
        for batch in loader_train:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)
            loss = calc_loss(out,batch.y,batch.D_sea,pars['train']['distance_to_sea'])
            loss.backward()

            optimizer.step()
            train_loss += loss.item()/d.n_train

        with torch.no_grad():
            for batch in loader_val:
                batch = batch.to(device)

                out = model(batch)
                loss = calc_loss(out,batch.y,batch.D_sea,pars['train']['distance_to_sea'])

                val_loss += loss.item()/d.n_val

        print(f'Epoch: {epoch}, Train_loss: {train_loss}, Validation_loss: {val_loss}')

        scheduler.step()

        ls.append(train_loss)
        ls_val.append(val_loss)

        if (pars['train']['reload_samples'] is not None) and ((epoch+1)%pars['train']['reload_samples']==0):
            print('Loading train samples')
            @parfor(range(d.n_train))
            def data_train(i):
                n = randintlog(pars['mesh']['n_min'],pars['mesh']['n_max'])
                return d.sample_graph(n, i, radius=pars['mesh']['radius'])

            loader_train = DataLoader(data_train, batch_size=pars['train']['batch_size'], shuffle=True)

        if (epoch+1)%10==0:
            memory_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
            save_model(model, pars, ls, ls_val, memory_use, start_time=start_time, seed=args.seed, comment=args.comment)

        if train_loss < loss_min:
            loss_min = train_loss
            loss_min_epoch = epoch
        elif epoch - loss_min_epoch > pars['train']['patience']:
            memory_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
            save_model(model, pars, ls, ls_val, memory_use, start_time=start_time, seed=args.seed, comment=args.comment)
            break