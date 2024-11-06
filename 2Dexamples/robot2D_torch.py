# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

class Robot2D:
    def __init__(self,
                 num_joints=3,
                 init_states=torch.tensor([[2*np.pi/3,-np.pi/3,-np.pi/3]]),
                 link_length=torch.tensor([[2,2,1]]).float(),
                 base_frame='default',
                 device='cpu',
                 ):
        self.device = device
        self.num_joints = num_joints
        self.init_states = init_states.to(self.device)
        self.link_length = link_length.to(self.device)
        self.B = init_states.size(0)
        if base_frame == 'default':
            self.base_frame = torch.zeros((self.B,2)).to(self.device)
        else:
            self.base_frame = base_frame.to(self.device)
        self.eef = self.forward_kinematics_eef(self.init_states)
        self.f_rob = self.forward_kinematics_all_joints(self.init_states)
        self.J = self.Jacobian(self.init_states)

    # Forward kinematics for end-effector (in robot coordinate system)
    def forward_kinematics_eef(self,x):
        self.B = x.size(0)
        L = torch.tril(torch.ones([self.num_joints, self.num_joints])).expand(self.B,-1,-1).float().to(self.device)
        x = x.unsqueeze(2)
        link_length = self.link_length.unsqueeze(1)
        f = torch.stack([
            torch.matmul(link_length, torch.cos(torch.matmul(L,x))),
            torch.matmul(link_length, torch.sin(torch.matmul(L,x)))
        ], dim=0).transpose(0,1).squeeze()
        if self.B ==1:
            f = f.unsqueeze(0)
        return f + torch.zeros((self.B,2)).to(self.device)

    # Forward kinematics for all joints (in robot coordinate system)
    def forward_kinematics_all_joints(self,x):
        self.B = x.size(0)
        L = torch.tril(torch.ones([self.num_joints,self.num_joints])).expand(self.B,-1,-1).float().to(self.device)
        x = x.unsqueeze(2)
        diag_length = torch.diag(self.link_length.squeeze()).unsqueeze(0)
        f = torch.stack([
            torch.matmul(L,torch.matmul(diag_length, torch.cos(torch.matmul(L,x)))),
            torch.matmul(L,torch.matmul(diag_length, torch.sin(torch.matmul(L,x))))
        ], dim=0).transpose(0,1).squeeze()
        if self.B ==1:
            f = f.unsqueeze(0)
        f = torch.cat([torch.zeros([self.B,2,1]).to(self.device), f], dim=-1)
        return f + torch.zeros((self.B,2)).to(self.device).unsqueeze(2).expand(-1, -1, self.num_joints + 1)

        # Forward kinematics for whole body (a is a parameter to control the position. a in [0,1])
    def forward_kinematics_any_point(self,x,a):
        ls = torch.sum(self.link_length)*a
        F = torch.zeros(self.B,2).to(self.device)
        f_rob = self.forward_kinematics_all_joints(x)
        for i,l in enumerate(ls):
            for j in range(self.num_joints):
                temp = self.link_length[0][j]
                if l > temp:
                    l = l - temp
                else:
                    f = f_rob[i,:,j] + (f_rob[i,:,j+1] - f_rob[i,:,j])*(l/temp)
                    F[i] = f
                    break
        return F

    # Jacobian with analytical computation (for single time step)
    def Jacobian(self,x):
        self.B = x.size(0)
        L = torch.tril(torch.ones([self.num_joints,self.num_joints])).expand(self.B,-1,-1).float().to(self.device)
        x = x.unsqueeze(2)
        diag_length = torch.diag(self.link_length.squeeze()).unsqueeze(0)
        J = torch.stack([
            torch.matmul(torch.matmul(-torch.sin(torch.matmul(L,x)).transpose(1,2), diag_length), L),
            torch.matmul(torch.matmul(torch.cos(torch.matmul(L,x)).transpose(1,2),diag_length), L),
            torch.ones(self.B,1,self.num_joints).to(self.device)
        ],dim=1).squeeze()
        if self.B ==1:
            J = J.unsqueeze(0)
        return J

    def surface_points_sampler(self,x,n=100):
        self.B = x.size(0)
        f_rob = self.forward_kinematics_all_joints(x) # B,2,N
        N = f_rob.size(2)
        t = torch.linspace(0,1,n).unsqueeze(0).expand(self.B,-1).to(self.device)
        kpts_list = []
        for i in range (N-1):
            # print(f_rob[:,:,i+1]-f_rob[:,:,i])
            kpts = torch.einsum('ij,ik->ijk',f_rob[:,:,i+1]-f_rob[:,:,i],t) + f_rob[:,:,i].unsqueeze(-1).expand(-1,-1,n)
            kpts_list.append(kpts)
        kpts = torch.cat(kpts_list,dim=-1).transpose(1,2)
        return kpts
    
    def distance(self,x,p):
        B = x.size(0)
        kpts = self.surface_points_sampler(x,n=200)
        Ns = kpts.size(1)
        p = p.unsqueeze(0).unsqueeze(2).expand(B,-1,Ns,-1)
        kpts = kpts.unsqueeze(1).expand(-1,p.size(1),-1,-1)
        dist = torch.norm(kpts-p,dim=-1).min(dim=-1)[0]
        return dist


if __name__ == "__main__":

    x = torch.tensor([[-1.6,-0.75]]) # Initial robot pose
    # x[1] = x[1] + np.pi
    rbt = Robot2D(num_joints=2,init_states = x,link_length=torch.tensor([[2,2]]).float())
    # a = torch.rand(5)
    # a.requires_grad =True
    # print(x.shape)
    # print(a.shape)
    # f = rbt.forward_kinematics_any_point(x,a)

    kpts = rbt.surface_points_sampler(x).numpy()
    # plt.plot(rbt.f_rob[0,0,:], rbt.f_rob[0,1,:], color=str(0), linewidth=1) # Plot robot
    plt.scatter(kpts[0,0,:], kpts[0,1,:], color=str(0), linewidth=1) # Plot robot
    plt.axis("equal")
    plt.show()

