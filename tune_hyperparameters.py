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

import wandb


if __name__ == '__main__':

    wandb.init()

    pars = dict()

    num_workers = 16

    # Data
    pars['data_path'] = 'data/era5_Santos_2022-2023.nc'
    pars['bath_path'] = 'data/era5_Santos_2022-2023_bath.nc'
    pars['train_frac'] = 0.7

    pars['mesh'] = {
        'n_min': int(0.5*wandb.config.mesh_n_max), # Minimum number of sample nodes in the domain
        'n_max': wandb.config.mesh_n_max, # Maximum number of sample nodes in the domain
        'radius': wandb.config.mesh_radius} # Maximum edge length in the graph

    # Model
    pars['model'] = {
        'width': wandb.config.model_width,
        'kernel_width': wandb.config.model_kernel_width,
        'depth': wandb.config.model_depth}

    # Training
    pars['train'] = {
        'distance_to_sea' : 300.,
        'batch_size': 10,
        'epochs': 100,
        'patience': 10,
        'learning_rate': 10**wandb.config.lr10,
        'scheduler_step': 100,
        'scheduler_gamma': 0.9}

    device = 'cuda'

    # -----------------------------------------
    # PRE-PROCESSING

    seed = 0
    comment = '-wandb'

    random.seed(seed)

    # Set up model
    model = KernelNN(pars['model']['width'], pars['model']['kernel_width'], pars['model']['depth'], 3, in_width=3, out_width=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pars['train']['learning_rate'], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=pars['train']['scheduler_step'], gamma=pars['train']['scheduler_gamma'])

    # Load data and generate meshes
    d = data_loader(pars['data_path'],pars['bath_path'],pars['train_frac'])

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

        if (epoch+1)%10==0:
            memory_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
            save_model(model, pars, ls, memory_use, start_time=start_time, seed=seed, comment=comment)

        wandb.log({
            'epoch': epoch, 
            'train_loss': train_loss, 
            'val_loss': val_loss
        })

        if train_loss < loss_min:
            loss_min = train_loss
            loss_min_epoch = epoch
        elif epoch - loss_min_epoch > pars['train']['patience']:
            memory_use = torch.cuda.max_memory_allocated(device)/(1024*1024)
            save_model(model, pars, ls, memory_use, start_time=start_time, seed=seed, comment=comment)
            break