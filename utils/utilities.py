import torch
import numpy as np
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from utils.nn_conv import NNConv_old
import torch.nn.functional
import netCDF4
import random
import time


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
        #self.U0 = np.stack((wind_u,wind_v,bath)) 

        y = np.array(ncdf_bath['latitude']) # Lat Bathymetry
        x = np.array(ncdf_bath['longitude']) # Lon Bathymetry

        # self.domain_boundaries = [self.x[0], self.x[-1], self.y[0], self.y[-1]]
        # Create mesh grid
        [self.y_grid, self.x_grid] = np.meshgrid(y,x)

        n_grid = n_lat*n_lon
        self.n_cases = n_cases

        self.x = self.x_grid.flatten()
        self.y = self.y_grid.flatten()
        self.wind_u = wind_u.transpose(1,2,0).reshape((n_grid,self.n_cases))
        self.wind_v = wind_v.transpose(1,2,0).reshape((n_grid,self.n_cases))
        self.bath = bath.flatten()
        self.shww = shww.transpose(1,2,0).reshape((n_grid,self.n_cases))

        is_land = self.bath < 0

        self.x = np.delete(self.x,is_land,axis=0)
        self.y = np.delete(self.y,is_land,axis=0)
        self.wind_u = np.delete(self.wind_u,is_land,axis=0)
        self.wind_v = np.delete(self.wind_v,is_land,axis=0)
        self.bath = np.delete(self.bath,is_land,axis=0)
        self.shww = np.delete(self.shww,is_land,axis=0)

        self.n_nodes = len(self.x)

        self.dist_norm = 1./((max(self.x) - min(self.x)) * np.pi/180. * self.radius)
        self.bath_norm = 1./999.
        self.wind_norm = 1./(np.std(self.wind_u[:,self.ind_train]))
        self.shww_norm = 1./(np.std(self.shww[:,self.ind_train]))


    def sample_nodes(self, n_points, n_sample, seed=None):
    # Gets a random sample of n_points points from sample number n_sample
    # Returns X and Y. Both with n_points lines
    # X is the input and has 4 columns, which are respectivelly x, y, U and U_prime (at the initial time)
    # Y is the output and has 2 columns, which are respectivelly U and U_prime (at the final time)

        if seed is not None:
            random.seed(seed)

        inds = random.sample(range(0,self.n_mesh), n_points)

        x = self.x_grid.flatten()[inds]
        y = self.y_grid.flatten()[inds]

        U0_1 = self.U0[:,:,0,n_sample].flatten()[inds]
        U0_2 = self.U0[:,:,1,n_sample].flatten()[inds]

        U1_1 = self.U1[:,:,0,n_sample].flatten()[inds]
        U1_2 = self.U1[:,:,1,n_sample].flatten()[inds]

        X = np.transpose(np.vstack((x,y,U0_1,U0_2)))
        Y = np.transpose(np.vstack((U1_1,U1_2)))

        return X, Y

    def sample_mesh(self, n_points, n_sample, radius=0, n_connections=0, seed=None, nodes=None):
    # Generates a mesh from a random sample
    # The neighborhood of each node is defined by a radius and by the number of connections, whichever creates the largest neighborhood
    # Returns a Torch Geometric Data object with the mesh

        # By default, the nodes are not provided, so we must first sample them
        if nodes is None:
            X, Y = self.sample_nodes(n_points, n_sample, seed=seed)
        else:
            X = nodes['X']
            Y = nodes['Y']
            n_points = X.shape[0]

        # Create mirrored version of domain: just one copy of each side is used as needing more is unrealistic
        X_left = [2*self.domian_boundaries[0], 0, 0, 0] + X*[-1, 1, 1, 1]
        X_right = [2*self.domian_boundaries[1], 0, 0, 0] + X*[-1, 1, 1, 1]
        X_middle = np.vstack((X,X_left,X_right))

        X_bottom = [0, 2*self.domian_boundaries[2], 0, 0] + X*[1, -1, 1, 1]
        X_top = [0, 2*self.domian_boundaries[3], 0, 0] + X*[1, -1, 1, 1]
        X_mirror = np.vstack((X_middle,X_bottom,X_top))

        # Compute pairwise distances
        dx = np.transpose([X[:,0]]) - X_mirror[:,0]
        dy = np.transpose([X[:,1]]) - X_mirror[:,1]
        D = np.sqrt(dx**2 + dy**2)

        edge_index = np.zeros([2,0])
        edge_attributes = np.zeros([1,0])

        # Iterate nodes
        for i in range(n_points):
            d = D[i,:]
            inds = np.argsort(d)
            if d[inds[n_connections]] > radius: # If n_connections creates the largest neighborhood
                inds = inds[0:n_connections]
            else: # If radius creates the largest neighborhood
                inds = inds[d[inds]<=radius]

            edge_index = np.hstack((edge_index, np.vstack((i*np.ones(len(inds)), inds%n_points)))) # Add index of edges to array
            edge_attributes = np.hstack((edge_attributes, d[inds].reshape(1,-1)))

        return Data(x=torch.tensor(X[:,[2,3]], dtype=torch.float),
                    y=torch.tensor(Y, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attributes, dtype=torch.float).t(),
                    norm_x=self.norm_x, norm_y=self.norm_y,
                    coords=X[:,[0,1]])#, norm_U = self.norm_U)

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