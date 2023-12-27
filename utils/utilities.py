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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class data_loader():
  
    def __init__(self, path, path_bath, train_frac=0.8):  
        ncdf = netCDF4.Dataset(path)
        ncdf_bath = netCDF4.Dataset(path_bath)
        n_cases = 2552
        n_lat = 51
        n_lon = 63
        self.radius = 6371000 # Earth radius to convert lat and lon to meters

        n_train = int(n_cases*train_frac)
        inds = list(range(n_cases))
        random.shuffle(inds)
        self.ind_train = inds[:n_train]
        self.ind_val = inds[n_train:]

        shww = np.array(ncdf['shww'])[:n_cases,0,:n_lat,:n_lon] #For some reason, goes only until 51,63. Lat and Lot seems to be wrong.

        wind_u = np.array(ncdf['u10'])[:n_cases,0,:n_lat,:n_lon]
        wind_v = np.array(ncdf['v10'])[:n_cases,0,:n_lat,:n_lon] 
        bath = np.array(ncdf_bath['wmb'])[0,0,:,:]  # Bathymetry

        y = np.array(ncdf_bath['latitude']) # Lat Bathymetry
        x = np.array(ncdf_bath['longitude']) # Lon Bathymetry

        # Create mesh grid
        [self.y_grid, self.x_grid] = np.meshgrid(y,x)

        n_grid = n_lat*n_lon
        self.n_cases = n_cases

        self.lon = self.x_grid.flatten()
        self.lat = self.y_grid.flatten()
        self.wind_u = wind_u.transpose(1,2,0).reshape((n_grid,self.n_cases))
        self.wind_v = wind_v.transpose(1,2,0).reshape((n_grid,self.n_cases))
        self.bath = bath.flatten()
        self.shww = shww.transpose(1,2,0).reshape((n_grid,self.n_cases))

        is_land = self.bath < 0

        self.lon = np.delete(self.lon,is_land,axis=0)
        self.lat = np.delete(self.lat,is_land,axis=0)
        self.wind_u = np.delete(self.wind_u,is_land,axis=0)
        self.wind_v = np.delete(self.wind_v,is_land,axis=0)
        self.bath = np.delete(self.bath,is_land,axis=0)
        self.shww = np.delete(self.shww,is_land,axis=0)

        self.n_nodes = len(self.lat)

        self.dist_norm = 1./((max(self.lat) - min(self.lat)) * np.pi/180. * self.radius)
        self.bath_norm = 1./999.
        self.wind_norm = 1./(np.std(self.wind_u[:,self.ind_train]))
        self.shww_norm = 1./(np.std(self.shww[:,self.ind_train]))


    def sample_graph(self, n_vertices, sample_n, radius=0, n_connections=0, validation=False, seed=None):
    # Samples a pair F and G from the dataset. Both functions are represented by graphs
    # Gets a random sample of n_vertices points from sample number sample_n from either the training or the validation data sets
    # F is a R2 -> R3 function, such that F = F(x,y) = [wind_u, wind_v, bath]^T
    # G is a R2 -> R1 function, such that G = G(x) = shww
    # Each vertice in the graph connects to the closest n_connections vertices within radius (whichever is larger)

        if seed is not None:
            random.seed(seed)

        if validation == False:
            case_ind = self.ind_train[sample_n]
        else:
            case_ind = self.ind_val[sample_n]

        vert_inds = random.sample(range(0,self.n_nodes), n_vertices)

        lon = self.lon[vert_inds]
        lat = self.lat[vert_inds]

        wind_u = self.wind_u[vert_inds,case_ind]*self.wind_norm
        wind_v = self.wind_v[vert_inds,case_ind]*self.wind_norm
        bath = self.bath[vert_inds]*self.bath_norm

        shww = self.shww[vert_inds,case_ind]*self.shww_norm

        F = np.stack((wind_u,wind_v,bath),axis=1)
        G = shww
        X = np.stack((lat,lon),axis=1)

        DS = pairwise_distances(np.radians(X), metric='haversine') * self.radius
        edge_index = np.zeros([0,2])
        edge_attributes = np.zeros([0,3])

        # Iterate nodes
        for i in range(n_vertices):
            ds = DS[i,:]
            inds = np.argsort(ds)
            if ds[inds[n_connections]] > radius: # If n_connections creates the largest neighborhood
                inds = inds[0:n_connections]
            else: # If radius creates the largest neighborhood
                inds = inds[ds[inds]<=radius]

            edge_index = np.vstack((edge_index, np.stack((i*np.ones(len(inds)), inds),axis=1))) # Add index of edges to array

            dx = self.radius * np.radians(lon[inds] - lon[i]) * np.cos(np.radians(lat[i]))
            dy = self.radius * np.radians(lat[inds] - lat[i])

            edge_attributes = np.vstack((edge_attributes, self.dist_norm*np.stack((ds[inds],dx,dy),axis=1)))


        return Data(F=torch.tensor(F, dtype=torch.float),
                    G=torch.tensor(G, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attributes, dtype=torch.float),
                    coords=X)

def spherical_distance(lat1,lat2,lon1,lon2,radius):
    dx = radius * (lon1-lon2)*np.cos(lon1)
    dy = radius * (lat1-lat2)
    ds = radius * np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon1-lon2))

    return dx, dy, ds

def save_model(model, pars, ls, memory_use, start_time=None, save_path=None, epoch=None, seed=None, comment=''):
    if epoch is None:
        epoch_text = ''
    else:
        epoch_text = f'_e{epoch}'

    if seed is None:
        seed_text = ''
    else:
        seed_text = f'_s{seed}'

    if save_path is None:
        save_path = f"models/model_nm{pars['mesh']['n_min']}_{pars['mesh']['n_max']}_r{pars['mesh']['radius']}_w{pars['model']['width']}_kw{pars['model']['kernel_width']}_d{pars['model']['depth']}{epoch_text}{seed_text}{comment}.pt"



    if start_time is None:
        run_time = None
    else:
        run_time = time.time()-start_time

    torch.save({'model': model.state_dict(), 'pars':pars, 'loss':ls, 'time':run_time, 'memory':memory_use}, save_path)

def randintlog(n1,n2):
    # Random integer between n1 and n2 with logarithmic distribution
    l1 = np.log(n1)
    l2 = np.log(n2)
    r = l1 + (l2-l1)*random.random()
    return(int(np.exp(r)))