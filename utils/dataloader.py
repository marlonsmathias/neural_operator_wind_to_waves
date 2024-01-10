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
  
    def __init__(self, path, path_bath, train_frac=0.8, seed=0, n_cases=None, lat_i=0, lat_f=41, lon_i=0, lon_f=63):  

        #n_cases = 2552

        # Set random seed
        if seed is not None:
            random.seed(seed)

        ncdf = netCDF4.Dataset(path)
        ncdf_bath = netCDF4.Dataset(path_bath)
        self.radius = 6371 # Earth radius to convert lat and lon to kilometers
        lat_i = 0
        lat_f = 41 # maximum of 51. 41 goes up to -30º
        lon_i = 0
        lon_f = 63
        total_cases = 2552

        if n_cases is None:
            n_cases = total_cases

        n_train = int(n_cases*train_frac)
        n_val = n_cases-n_train

        inds = list(range(total_cases))
        random.shuffle(inds)
        self.ind_train = inds[:n_train]
        self.ind_val = inds[n_train:n_train+n_val]

        self.n_train = n_train
        self.n_val = n_val

        shww = np.array(ncdf['shww'])[:total_cases,0,lat_i:lat_f,lon_i:lon_f] #For some reason, goes only until 51,63. Lat and Lot seems to be wrong.

        wind_u = np.array(ncdf['u10'])[:total_cases,0,lat_i:lat_f,lon_i:lon_f]
        wind_v = np.array(ncdf['v10'])[:total_cases,0,lat_i:lat_f,lon_i:lon_f] 
        bath = np.array(ncdf_bath['wmb'])[0,0,lat_i:lat_f,lon_i:lon_f]  # Bathymetry

        y = np.array(ncdf_bath['latitude'])[lat_i:lat_f] # Lat
        x = np.array(ncdf_bath['longitude'])[lon_i:lon_f] # Lon 

        # Create mesh grid
        [self.y_grid, self.x_grid] = np.meshgrid(y,x)


        n_grid = (lat_f-lat_i)*(lon_f-lon_i)
        self.n_cases = n_cases

        self.lon = self.x_grid.flatten()
        self.lat = self.y_grid.flatten()
        self.wind_u = wind_u.transpose(2,1,0).reshape((n_grid,total_cases))
        self.wind_v = wind_v.transpose(2,1,0).reshape((n_grid,total_cases))
        self.bath = bath.transpose(1,0).flatten()
        self.shww = shww.transpose(2,1,0).reshape((n_grid,total_cases))

        is_land = self.bath < 0

        # Identify which points are next to open sea outside the domain
        open_sea = (self.bath == max(self.bath)) * ((self.lon == max(self.lon)) + (self.lon == min(self.lon)) + (self.lat == max(self.lat)) + (self.lat == min(self.lat)))
        self.sea_coords = np.stack((self.lat[open_sea],self.lon[open_sea]),axis=1)

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

        #F = np.stack((wind_u,wind_v,bath,lon,lat),axis=1)
        F = np.stack((wind_u,wind_v,bath),axis=1)
        G = shww
        X = np.stack((lat,lon),axis=1)

        DS = pairwise_distances(np.radians(X), metric='haversine') * self.radius

        # Distance from each point to open sea
        D_sea = np.min(pairwise_distances(np.radians(X),np.radians(self.sea_coords), metric='haversine'),axis=1) * self.radius

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


        return Data(x=torch.tensor(F, dtype=torch.float),
                    y=torch.tensor(G, dtype=torch.float),
                    D_sea = torch.tensor(D_sea, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long).t(),
                    edge_attr=torch.tensor(edge_attributes, dtype=torch.float),
                    coords=X)