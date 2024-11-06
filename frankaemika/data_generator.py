# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import torch
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR,'../../RDF'))
from panda_layer.panda_layer import PandaLayer
from bf_sdf import BPSDF
from torchmin import minimize
import time
import math
import copy

PI = math.pi

class DataGenerator():
    def __init__(self,device):
        # panda model
        self.panda = PandaLayer(device)
        self.bp_sdf = BPSDF(8,-1.0,1.0,self.panda,device)
        self.model = torch.load(os.path.join(CUR_DIR,'../../RDF/models/BP_8.pt'))
        self.q_max = self.panda.theta_max
        self.q_min = self.panda.theta_min
        # device
        self.device = device

        # data generation
        self.workspace = [[-0.5,-0.5,0.0],[0.5,0.5,1.0]]
        self.n_disrete = 20         # total number of x: n_discrete**3
        self.batchsize = 20000       # batch size of q
        # self.pose = torch.eye(4).unsqueeze(0).to(self.device).expand(self.batchsize,4,4).float()
        self.epsilon = 1e-3         # distance threshold to filter data

    def compute_sdf(self,x,q,return_index = False):
        # x : (Nx,3)
        # q : (Nq,7)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq)
        # return idx : (Nq) optional

        pose = torch.eye(4).unsqueeze(0).to(self.device).expand(len(q),4,4).float()
        if not return_index:
            d,_ = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.model,use_derivative =False)
            d = d.min(dim=1)[0]
            return d
        else:
            d,_,idx = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.model,use_derivative =False,return_index = True)
            d,pts_idx = d.min(dim=1)
            idx = idx[torch.arange(len(idx)),pts_idx]
            return d,idx

    def given_x_find_q(self,x,q = None, batchsize = None,return_mask = False,epsilon = 1e-3):
        # x : (N,3)
        # scale x to workspace
        if not batchsize:
            batchsize = self.batchsize

        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2
            # x : N,3

            d = self.compute_sdf(x,q)
            cost = torch.sum(d**2)
            return cost
        
        t0 = time.time()
        # optimizer for data generation
        if q is None:
            q = torch.rand(batchsize,7).to(self.device)*(self.q_max-self.q_min)+self.q_min
        q0 = copy.deepcopy(q)
        res = minimize(
            cost_function, 
            q, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
            )
        
        d,idx = self.compute_sdf(x,res.x,return_index=True)
        d,idx = d.squeeze(),idx.squeeze()

        mask = torch.abs(d) < epsilon
        # q_valid,d,idx = res.x[mask],d[mask],idx[mask]
        boundary_mask = ((res.x > self.q_min) & (res.x < self.q_max)).all(dim=1)
        final_mask = mask & boundary_mask
        final_q,idx = res.x[final_mask],idx[final_mask]
        # q0 = q0[mask][boundary_mask]

        print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q),time.time()-t0))
        if return_mask:
            return final_mask,final_q,idx
        else:
            return final_q,idx

    def distance_q(self,x,q):
        # x : (Nx,3)
        # q : (Np,7)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0

        # compute d
        Np = q.shape[0]
        q_template,link_idx = self.given_x_find_q(x)
        print(q_template.shape)

        if link_idx.min() == 0:
            return torch.zeros(Np).to(self.device)
        else:
            link_idx[link_idx==7] = 6
            link_idx[link_idx==8] = 7
            d = torch.inf*torch.ones(Np,7).to(self.device)
            for i in range(link_idx.min(),link_idx.max()+1):
                mask = (link_idx==i)
                d_norm = torch.norm(q[:,:i].unsqueeze(1)- q_template[mask][:,:i].unsqueeze(0),dim=-1)
                d[:,i-1] = torch.min(d_norm,dim=-1)[0]
        d = torch.min(d,dim=-1)[0]

        # compute sign of d
        d_ts = self.compute_sdf(x,q)
        mask =  (d_ts < 0)
        d[mask] = -d[mask]
        return d 

    def projection(self,x,q):
        q.requires_grad = True
        d = self.distance_q(x,q)
        grad = torch.autograd.grad(d,q,torch.ones_like(d),create_graph=True)[0]
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def generate_offline_data(self,save_path = CUR_DIR):
        
        x = torch.linspace(self.workspace[0][0],self.workspace[1][0],self.n_disrete).to(self.device)
        y = torch.linspace(self.workspace[0][1],self.workspace[1][1],self.n_disrete).to(self.device)
        z = torch.linspace(self.workspace[0][2],self.workspace[1][2],self.n_disrete).to(self.device)
        x,y,z = torch.meshgrid(x,y,z)
        pts = torch.stack([x,y,z],dim=-1).reshape(-1,3)
        data = {}
        for i,p in enumerate(pts):
            q,idx = self.given_x_find_q(p.unsqueeze(0)) 
            data[i] ={
                'x':    p.detach().cpu().numpy(),
                'q':    q.detach().cpu().numpy(),
                'idx':  idx.detach().cpu().numpy(),
            }
            print(f'point {i} finished, number of q: {len(q)}')
        # np.save(os.path.join(save_path,'data.npy'),data)

def analysis_data(x):
    # Compute the squared Euclidean distance between each row
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    # Set the diagonal elements to a large value to exclude self-distance
    diag_indices = torch.arange(x.shape[0])
    diff[diag_indices, diag_indices] = float('inf')
    
    # Compute the Euclidean distance by taking the square root
    diff = diff.sqrt()
    min_dist = torch.min(diff,dim=1)[0]
    print(f'distance\tmax:{min_dist.max()}\tmin:{min_dist.min()}\taverage:{min_dist.mean()}')



if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gen = DataGenerator(device)
    # x = torch.tensor([[0.5,0.5,0.5]]).to(device)
    # gen.single_point_generation(x)
    gen.generate_offline_data()