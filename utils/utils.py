import torch
import numpy as np
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from utils.nn_conv import NNConv_old
import torch.nn.functional
import netCDF4
import random
import time
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F

def save_model(model, pars, ls, ls_val, memory_use, start_time=None, save_path=None, epoch=None, seed=None, comment=''):
    if epoch is None:
        epoch_text = ''
    else:
        epoch_text = f'_e{epoch}'

    if seed is None:
        seed_text = ''
    else:
        seed_text = f'_s{seed}'

    if pars['train']['n_cases'] is None:
        ncases_text = ''
    else:
        ncases_text = f'_nc{pars["train"]["n_cases"]}'

    if pars['train']['reload_samples'] is None:
        reload_text = ''
    else:
        reload_text = f'_rl{pars["train"]["reload_samples"]}'

    if save_path is None:
        save_path = f"models/model_nm{pars['mesh']['n_min']}_{pars['mesh']['n_max']}_ds{pars['train']['distance_to_sea']}_r{pars['mesh']['radius']}_w{pars['model']['width']}_kw{pars['model']['kernel_width']}_d{pars['model']['depth']}{ncases_text}{reload_text}{epoch_text}{seed_text}{comment}.pt"



    if start_time is None:
        run_time = None
    else:
        run_time = time.time()-start_time

    torch.save({'model': model.state_dict(), 'pars':pars, 'loss':ls, 'loss_val':ls_val, 'time':run_time, 'memory':memory_use}, save_path)

def randintlog(n1,n2):
    # Random integer between n1 and n2 with logarithmic distribution
    l1 = np.log(n1)
    l2 = np.log(n2)
    r = l1 + (l2-l1)*random.random()
    return(int(np.exp(r)))

def calc_loss(P,T,dist,dist_min):
    P_inner = P.view(-1, 1)[dist > dist_min]
    T_inner = T.view(-1, 1)[dist > dist_min]
    return F.mse_loss(P_inner, T_inner)