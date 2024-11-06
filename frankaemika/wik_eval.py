# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import numpy as np
import os
import sys
import torch
import matplotlib.pyplot as plt
import time
import math
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_PATH,'../../RDF'))
from mlp import MLPRegression
from panda_layer.panda_layer import PandaLayer
import bf_sdf
from nn_cdf import CDF
import copy
from data_generator import DataGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# initialize CDF
cdf = CDF(device)
model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
model.to(device)

# initialize bp_sdf model 
bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,cdf.panda,device)
bp_sdf_model = torch.load(os.path.join(CUR_PATH,'../../RDF/models/BP_8.pt'))

# initialize njsdf model
sys.path.append(os.path.join(CUR_PATH,'../../Neural-JSDF/learning/nn-learning'))
from sdf.robot_sdf import RobotSdfCollisionNet
from torchmin import minimize

tensor_args = {'device': device, 'dtype': torch.float32}
njsdf_model = RobotSdfCollisionNet(in_channels=10, out_channels=9, layers=[256] * 4, skips=[])
njsdf_model.load_weights('../../Neural-JSDF/learning/nn-learning/sdf_256x5_mesh.pt', tensor_args)
njsdf_model = njsdf_model.model

def find_q_njsdf(x,q=None,batchsize = 10000):
    # x : (N,3)
    # scale x to workspace

    def compute_njsdf(x,q):
        q = q.unsqueeze(1).expand(len(q),len(x),7)
        x = x.unsqueeze(0).expand(len(q),len(x),3)
        x_cat = torch.cat([q,x],dim=-1).float().reshape(-1,10)
        dist = njsdf_model.forward(x_cat)/100.
        d = torch.min(dist[:,:8],dim=1)[0]
        return d

    def cost_function(q):
        #  find q that d(x,q) = 0
        # q : B,2
        # x : N,3
        d = compute_njsdf(x,q)
        cost = torch.sum(d**2)
        return cost
    
    t0 = time.time()
    # optimizer for data generation
    if q is None:
        q = torch.rand(batchsize,7).to(device)*(cdf.panda.theta_max-cdf.panda.theta_min)+cdf.panda.theta_min
    q0 = copy.deepcopy(q)
    res = minimize(
        cost_function, 
        q, 
        method='l-bfgs', 
        options=dict(line_search='strong-wolfe'),
        max_iter=50,
        disp=0
        )
    
    return q0,res.x

dg = DataGenerator(device)



def eval_main():
    # method_list = ['SDF','NJSDF','CDF']

    method = 'CDF'
    time_list, num_solutions, avg_dist = [],[],[]
    if method == 'CDF':
        for t in range (100):
            x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
            t0 = time.time()
            q = cdf.sample_q(batch_q=10000)
            q0 = copy.deepcopy(q)
            for _ in range (2):
                d,grad = cdf.inference_d_wrt_q(x,q,model,return_grad = True)
                q = cdf.projection(q,d,grad)
            q,grad = q.detach(),grad.detach()   # release memory
            pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(cdf.device).float()
            sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
            
            error = sdf.reshape(-1).abs()
            mask = error < 0.03
            q = q[mask]
            dist = torch.norm(q-q0[mask],dim=-1).mean().item()
            print('iter:',t, 'avg dist: ',dist)
            if t>0: 
                time_list.append(time.time()-t0)
                num_solutions.append(len(q))
                avg_dist.append(dist)
        print('average time cost: ',np.mean(time_list),'num solutions: ',np.mean(num_solutions),'avg dist: ',np.mean(avg_dist))

    if method == 'SDF':
        for t in range (1000):
            x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
            t0 = time.time()
            mask,q,idx = dg.given_x_find_q(x,batchsize=1000,return_mask=True,epsilon=0.03)
            # pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(cdf.device).float()
            # sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
            
            # error = sdf.reshape(-1).abs()
            # mask = error < 0.03
            if t>0: 
                time_list.append(time.time()-t0)
                num_solutions.append(len(q))
                avg_dist.append(dist)
        print('average time cost: ',np.mean(time_list),'num solutions: ',np.mean(num_solutions))

    if method == 'NJSDF':
        for t in range (100):
            x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
            t0 = time.time()
            q0,q = find_q_njsdf(x)
            pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(cdf.device).float()
            sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
            
            error = sdf.reshape(-1).abs()
            mask = error < 0.03
            q0,q = q0[mask],q[mask]
            dist = torch.norm(q-q0,dim=-1).mean().item()
            if t>0: 
                time_list.append(time.time()-t0)
                num_solutions.append(len(q))
                avg_dist.append(dist)
        print('average time cost: ',np.mean(time_list),'num solutions: ',np.mean(num_solutions),'avg dist: ',np.mean(avg_dist))

def eval_dist():
    batchsize = 10000
    d_list = []
    for t in range(100):
        x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
        q = torch.torch.rand(batchsize,7).to(device)*(cdf.panda.theta_max-cdf.panda.theta_min)+cdf.panda.theta_min
        q0 = copy.deepcopy(q)
        # CDF
        q.requires_grad = True
        for i in range (3):
            d,grad = cdf.inference_d_wrt_q(x,q,model,return_grad = True)
            q = cdf.projection(q,d,grad)
            if i == 0:
                q_step1 = q.clone()
            if i == 1:
                q_step2 = q.clone()
            if i ==2:
                q_step3 = q.clone()

        q_CDF1 = q_step1.clone()
        q_CDF2 = q_step2.clone()
        q_CDF3 = q_step3.clone()

        # SDF
        mask,q_sdf,_ = dg.given_x_find_q(x,q = q0,batchsize=batchsize,return_mask=True,epsilon=0.03)
        # Neural JSDF
        q0,q_njsdf = find_q_njsdf(x,q=q0)
        d_CDF1 = torch.norm(q_CDF1[mask]-q0[mask],dim=-1).mean()
        d_CDF2 = torch.norm(q_CDF2[mask]-q0[mask],dim=-1).mean()
        d_CDF3 = torch.norm(q_CDF3[mask]-q0[mask],dim=-1).mean()
        d_sdf = torch.norm(q_sdf-q0[mask],dim=-1).mean()
        d_njsdf = torch.norm(q_njsdf[mask]-q0[mask],dim=-1).mean()
        d_list.append([d_CDF1.item(),d_CDF2.item(),d_CDF3.item(),d_sdf.item(),d_njsdf.item()])
        print('iter:',t,'d_CDF:',d_CDF1.item(),d_CDF2.item(),d_CDF3.item(),'d_sdf:',d_sdf.item(),'d_njsdf:',d_njsdf.item())
    d_list = np.array(d_list)
    print('d_CDF1:',np.mean(d_list[:,0]),'d_CDF2:',np.mean(d_list[:,1]),'d_CDF3:',np.mean(d_list[:,2]),'d_sdf:',np.mean(d_list[:,3]),'d_njsdf:',np.mean(d_list[:,4]))


# eval_dist()
eval_main()

