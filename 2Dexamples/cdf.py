# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

# 2D example for configuration space distance field
import numpy as np
import os
import sys
import torch
import math
import matplotlib.pyplot as plt
from robot2D_torch import Robot2D
from primitives2D_torch import Circle
from torchmin import minimize
import time
import math
import robot_plot2D
import copy
import matplotlib.gridspec as gridspec

PI = math.pi
CUR_PATH = os.path.dirname(os.path.realpath(__file__))

class CDF2D:
    def __init__(self,device) -> None:
        self.device = device    
        self.nbData =  100
        self.nbDiscretization = 50
        self.Q_grid = self.create_grid_torch(self.nbData).to(device)
        self.link_length = torch.tensor([[2,2]]).float().to(device)
        self.num_joints = self.link_length.size(1)
        self.q_max = torch.tensor([PI,PI]).to(device)
        self.q_min = torch.tensor([-PI,-PI]).to(device)

        # data generation
        self.task_space = [[-3.0,-3.0],[3.0,3.0]]
        self.batchsize = 40000       # batch size of q
        self.epsilon = 1e-3         # distance threshold to filter data

        # robot
        self.robot = Robot2D(num_joints=self.num_joints ,init_states = self.Q_grid,link_length=self.link_length,device = device)

        # c space distance field
        if not os.path.exists(os.path.join(CUR_PATH,'data2D.pt')):
            self.generate_data()
        self.q_grid_template =  torch.load(os.path.join(CUR_PATH,'data2D.pt'))

    def create_grid(self,nb_data):
        t = np.linspace(-math.pi,math.pi, nb_data)
        self.q0,self.q1 = np.meshgrid(t,t)
        return self.q0,self.q1
    
    def create_grid_torch(self,nb_data):
        q0,q1 = self.create_grid(nb_data)
        q0_torch = torch.from_numpy(q0).float()
        q1_torch = torch.from_numpy(q1).float()
        Q_sets = torch.cat([q0_torch.unsqueeze(-1),q1_torch.unsqueeze(-1)],dim=-1).view(-1,2)
        return Q_sets

    def inference_sdf(self,q,obj_lists,return_grad = False):  
        # using predefined object 
        kpts = self.robot.surface_points_sampler(q)
        B,N = kpts.size(0),kpts.size(1)
        dist = torch.cat([obj.signed_distance(kpts.reshape(-1,2)).reshape(B,N,-1) for obj in obj_lists],dim=-1)

        # using closest point from robot surface
        sdf = torch.min(dist,dim=-1)[0]
        sdf = sdf.min(dim=-1)[0]
        if return_grad: 
            grad = torch.autograd.grad(sdf,q,torch.ones_like(sdf))[0]
            return sdf,grad
        return sdf
    
    def find_q(self,obj_lists,batchsize = None):
        # find q that makes d(x,q) = 0. x is the obstacle surface
        # using L-BFGS method
        if not batchsize:
            batchsize = self.batchsize
            
        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,2

            d = self.inference_sdf(q,obj_lists)
            cost = torch.sum(d**2)
            return cost
        
        t0 = time.time()
        # optimizer for data generation
        q = torch.rand(batchsize,2).to(self.device)*(self.q_max-self.q_min)+self.q_min
        q0 =copy.deepcopy(q)
        res = minimize(
            cost_function, 
            q, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=50,
            disp=0
            )
        
        d = self.inference_sdf(q,obj_lists).squeeze()

        mask = torch.abs(d) < 0.05
        q_valid,d = res.x[mask],d[mask]
        boundary_mask = ((q_valid > self.q_min) & (q_valid < self.q_max)).all(dim=1)
        final_q = q_valid[boundary_mask]
        q0 = q0[mask][boundary_mask]
        # print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q),time.time()-t0))
        return q0,final_q,res.x
    
    def generate_data(self,nbDiscretization=50):
        x = torch.linspace(self.task_space[0][0],self.task_space[1][0],self.nbDiscretization).to(self.device)
        y = torch.linspace(self.task_space[0][1],self.task_space[1][1],self.nbDiscretization).to(self.device)
        xx,yy = torch.meshgrid(x,y)
        xx,yy = xx.reshape(-1,1),yy.reshape(-1,1)
        p = torch.cat([xx,yy],dim=-1).to(self.device)

        data = {}
        for i,_p in enumerate(p):
            grids = [Circle(center=_p,radius=0.001,device=device)]
            q = self.find_q(grids)[1]
            data[i] = {
                'p':_p,
                'q':q
            }
            print('i: {} \t number of q: {}'.format(i,len(q[1])))
        np.save(os.path.join(CUR_PATH,'data2D.npy'),data)
        data = np.load(os.path.join(CUR_PATH,'data2D.npy'),allow_pickle=True).item()
        max_q_per_x = 200
        tensor_data = torch.inf*torch.ones(self.nbData,self.nbData,max_q_per_x,2).to(self.device)
        for idx in data.keys():
            p = data[idx]['p']
            q = data[idx]['q']
            i = idx/50
            j = idx%50
            if len(q) > max_q_per_x:
                q = q[:max_q_per_x]
            tensor_data[int(i),int(j),:len(q),:] = q
        torch.save(tensor_data,os.path.join(CUR_PATH,'data2D.pt'))
        return tensor_data

    def calculate_cdf(self,q,obj_lists,method='online_computation',return_grad = False):
        # x : (Nx,2)
        # q : (Np,2)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0
        Np = q.shape[0]
        if method == None:
            method = 'online_computation'
        if method == 'offline_grid':
            if not hasattr(self,'q_list_template'):
                obj_points = torch.cat([obj.sample_surface(200) for obj in obj_lists])
                grid = self.x_to_grid(obj_points)      
                q_list_template = (self.q_grid_template[grid[:,0],grid[:,1],:,:]).reshape(-1,2)
                self.q_list_template = q_list_template[q_list_template[:,0] != torch.inf]
            dist = torch.norm(q.unsqueeze(1) - self.q_list_template.to(self.device).unsqueeze(0),dim=-1)
        if method == 'online_computation':
            if not hasattr(self,'q_0_level_set'):
                self.q_0_level_set = self.find_q(obj_lists)[1]
            dist = torch.norm(q.unsqueeze(1) - self.q_0_level_set.unsqueeze(0),dim=-1)
  
        d = torch.min(dist,dim=-1)[0]
        # compute sign of d, based on the sdf
        # exit()
        d_ts = self.inference_sdf(q,obj_lists)

        mask =  (d_ts < 0)
        d[mask] = -d[mask]
        if return_grad:
            grad = torch.autograd.grad(d,q,torch.ones_like(d))[0]
            return d,grad
        return d 
    
    def x_to_grid(self,p):
        # p: (N,2)
        # return grid index (N,2)
        x_workspace = torch.tensor([self.task_space[0][0],self.task_space[1][0]]).to(self.device)
        y_workspace = torch.tensor([self.task_space[0][1],self.task_space[1][1]]).to(self.device)

        x_grid = (p[:,0]-x_workspace[0])/(x_workspace[1]-x_workspace[0])*self.nbDiscretization
        y_grid = (p[:,1]-y_workspace[0])/(y_workspace[1]-y_workspace[0])*self.nbDiscretization

        x_grid.clamp_(0,self.nbDiscretization-1)
        y_grid.clamp_(0,self.nbDiscretization-1)
        return torch.stack([x_grid,y_grid],dim=-1).long()
    
    def projection(self,q,d,grad):
        # q : (N,2)
        # d : (N)
        # grad : (N,2)
        # return q_proj : (N,2)
        q_proj = q - grad*d.unsqueeze(-1)
        return q_proj
    
    def plot_projection(self,ax):
        q = torch.rand(1000,2).to(self.device)*2*math.pi-math.pi
        q0 = copy.deepcopy(q)
        d,grad = self.inference_c_space_sdf_using_data(q,sign=False)
        q = self.projection(q,d,grad)
        q,grad= q.detach(),grad.detach()   # release memory
        ax.set_title(f'{iter} iteractions', size=25)  # Add a title to your plot
        ax.plot(q[:,0].detach().cpu().numpy(),q[:,1].detach().cpu().numpy(),'.',color='lightgreen')
        return q0,q

    def plot_sdf(self,obj_lists, ax):
        ax.set_aspect('equal', 'box')  # Make sure the pixels are square
        ax.set_title('Configuration space', size=30)  # Add a title to your plot
        ax.set_xlabel('q1', size=20)
        ax.set_ylabel('q2', size=20)
        axis_limits = (-PI, PI)  # Set the limits for both axes to be the same
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.tick_params(axis='both', labelsize=20)

        sdf = self.inference_sdf(self.Q_grid,obj_lists)
        sdf = sdf.detach().cpu().numpy()

        ax.contour(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=[0], linewidths=6, colors='black', alpha=1.0)
        ct = ax.contourf(self.q0, self.q1, sdf.reshape(self.nbData, self.nbData), levels=6, linewidths=1, cmap='coolwarm')
        ax.clabel(ct, inline=False, fontsize=15, colors='black', fmt='%.1f')

        # fig = plt.gcf()  # Get the current figure
        # fig.colorbar(ct, ax=ax)  # Add a colorbar to your plot

    def shooting(self,q0,obj_lists,dt = 1e-2,timestep = 500,method = 'SDF'):
        q = q0
        q.requires_grad = True
        q_list = []
        for t in range(timestep):
            if method == 'SDF':
                d,g = self.inference_sdf(q,obj_lists, return_grad=True)
            if method == 'CDField':
                d,g = self.calculate_cdf(q,obj_lists,return_grad=True)
            g = torch.nn.functional.normalize(g,dim=-1)
            g_orth = torch.stack([g[:,1],-g[:,0]],dim=-1)
            # if g_orth[:,1] < 0:
            #     g_orth = -g_orth
            q = q + dt*g_orth
            q_list.append(q.detach().cpu().numpy())
        return np.array(q_list).transpose(1,0,2)
    
    def shooting_proj(self,q0,obj_lists,dt = 1e-2,timestep = 500,method = 'SDF'):
        q = q0
        q.requires_grad = True
        q_list = []
        if method == 'SDF':
            for t in range(timestep):
                q_list.append(q.detach().cpu().numpy())
                d,g = self.inference_sdf(q,obj_lists,return_grad=True)
                # if method == 'CDField':
                #     d,g = self.calculate_cdf(q,obj_lists,return_grad=True)
                g = torch.nn.functional.normalize(g,dim=-1)
                # if g_orth[:,1] < 0:
                #     g_orth = -g_orth
                q = q - dt*g*d.unsqueeze(-1)
                # q_list.append(q.detach().cpu().numpy())
        if method == 'CDField':
            d,g = self.calculate_cdf(q,obj_lists,return_grad=True)
            q = q - g*d.unsqueeze(-1)
            q_list = np.linspace(q0.detach().cpu().numpy(),q.detach().cpu().numpy(),timestep)
        return np.array(q_list).transpose(1,0,2)

    def plot_cdf(self,ax,obj_lists,method='online_computation'):
        d = self.calculate_cdf(self.Q_grid,obj_lists,method).detach().cpu().numpy()
        ax.set_aspect('equal', 'box')  # Make sure the pixels are square
        ax.set_title('Configuration space', size=30)  # Add a title to your plot
        ax.set_xlabel('q1', size=20)
        ax.set_ylabel('q2', size=20)
        axis_limits = (-PI, PI)  # Set the limits for both axes to be the same
        ax.set_xlim(axis_limits)
        ax.set_ylim(axis_limits)
        ax.tick_params(axis='both', labelsize=20)

        ax.contour(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=[0], linewidths=6, colors='black', alpha=1.0)
        ct = ax.contourf(self.q0, self.q1, d.reshape(self.nbData, self.nbData), levels=8, linewidths=1, cmap='coolwarm')
        ax.clabel(ct, inline=False, fontsize=15, colors='black', fmt='%.1f')


    def plot_objects(self,ax,obj_lists):
        for obj in obj_lists:
            # plt.gca().add_patch(obj.create_patch())
            ax.add_patch(obj.create_patch())
        return ax
    
    def compare_with_lbfgs(self):
        q0,q,_ = self.find_q()
        return q0,q
    
def plot_fig1(obj_lists):
    color_list = ['magenta','orange', 'cyan', 'green',  'purple', 'navy']
    fig1, ax1 = plt.subplots(figsize=(10,8))  # Create the first plot
    fig2, ax2 = plt.subplots(figsize=(10, 8))  # Create the third plot

    fig = plt.figure(figsize=(25,20))  # Create a figure
    gs = gridspec.GridSpec(2, 6)  # Create a gridspec
    xlim=(-4.0,4.0)
    ylim=(-4.0,4.0)
    axs = []
    for i in range(12):
        ax = fig.add_subplot(gs[i // 6, i % 6])  # Add a subplot
        ax.axis('off')  # Turn off the axis
        cdf.plot_objects(ax,obj_lists)
        ax.set_aspect('equal', 'box')  # Make sure the pixels are square
        ax.set_xlim(xlim)  # Set the x limits
        ax.set_ylim(ylim)  # Set the y limits
        axs.append(ax)

    # Plot sdf on the first subplot
    import matplotlib.cm as cm
    cdf.plot_sdf(ax=ax1,obj_lists=obj_lists)
    # plot shooting
    shooting_q0 = torch.tensor([[-0.2,0.0],[1.2,1.0],[-1.0,0.5]]).to(device)
    projection_q0 = torch.tensor([[0.0,-1.5],[2.0,2.0],[-1.5,-2.0]]).to(device)
    shooting_tangent = cdf.shooting(shooting_q0,obj_lists,method='SDF')
    shooting_gradient = cdf.shooting_proj(projection_q0,obj_lists,method='SDF')
    shooting_q_sdf = np.concatenate([shooting_gradient,shooting_tangent],axis=0)    
    for c,shoot in enumerate(shooting_q_sdf):
        ax1.plot(shoot[:,0],shoot[:,1],'r--',color = color_list[c],linewidth=3)
        ax1.plot(shoot[0,0],shoot[0,1],'*',color = color_list[c],markersize=10)
        # plot robot
        robot_plot2D.plot_2d_manipulators(joint_angles_batch=shoot[0:500:20],ax = axs[c*2],color = color_list[c],show_start_end=False,show_eef_traj=True)
    cdf.plot_cdf(ax2,obj_lists)

    shooting_tangent = cdf.shooting(shooting_q0,obj_lists,method='CDField')
    shooting_gradient = cdf.shooting_proj(projection_q0,obj_lists,method='CDField')
    shooting_q_cdf = np.concatenate([shooting_gradient,shooting_tangent],axis=0)    
    for c,shoot in enumerate(shooting_q_cdf):
        ax2.plot(shoot[:,0],shoot[:,1],'r--',color = color_list[c],linewidth=3)
        ax2.plot(shoot[0,0],shoot[0,1],'*',color = color_list[c],markersize=10)
        if c<3:
            # gradient projection
            robot_plot2D.plot_2d_manipulators(joint_angles_batch=shoot,ax = axs[c*2+1],color = color_list[c],show_start_end=True,show_eef_traj=True) 
        else:
            # geodesic shooting
            robot_plot2D.plot_2d_manipulators(joint_angles_batch=shoot[0:500:20],ax = axs[c*2+1],color = color_list[c],show_start_end=False,show_eef_traj=True) 
    plt.show()

def plot_projection(obj_lists):
    fig1,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(24,8))

    # plot cdf
    cdf.plot_cdf(ax1,obj_lists)
    cdf.plot_cdf(ax2,obj_lists)
    q_random = torch.rand(1000,2,requires_grad=True).to(device)*2*math.pi-math.pi
    
    # plot CDF
    d,grad = cdf.calculate_cdf(q_random,obj_lists,return_grad=True)
    q_proj = cdf.projection(q_random,d,grad)
    ax1.plot(q_random[:,0].detach().cpu().numpy(),q_random[:,1].detach().cpu().numpy(),'.',color='lightgreen')
    ax2.plot(q_proj[:,0].detach().cpu().numpy(),q_proj[:,1].detach().cpu().numpy(),'.',color='lightgreen')
    ax1.set_title('Initial Samples', size=25)  # Add a title to your plot
    ax2.set_title('CDF + Gradient Projection', size=25)  # Add a title to your plot

    # plot SDF for comparison
    cdf.plot_sdf(obj_lists,ax3)
    q0,q,q_all = cdf.find_q(obj_lists,batchsize=1000)
    ax3.set_title(f'SDF+Optimization', size=25)  # Add a title to your plot
    ax3.plot(q_all[:,0].detach().cpu().numpy(),q_all[:,1].detach().cpu().numpy(),'.',color='lightgreen')

    # plot robot manipulator in task space for CDF
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure 
    q_proj_np = q_proj.detach().cpu().numpy()
    for i, _q in enumerate(q_proj_np):
        robot_plot2D.plotArm(
                ax=ax,
                a=_q,
                d=cdf.link_length[0].cpu().numpy(),
                p=np.array([0.0, 0.0]),  # base position
                sz=0.05,
                label="via",
                alpha=0.05,
                zorder=2,
                xlim=None,
                ylim=None,
                robot_base=True,  # to visualize the base
                color='lightgreen'  # Set the color of the robot
            )
    cdf.plot_objects(ax,obj_lists)

    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cdf = CDF2D(device)

    scene_1 = [Circle(center=torch.tensor([2.5,2.5]),radius=0.5,device=device)]
    scene_2 = [Circle(center=torch.tensor([2.3,-2.3]),radius=0.3,device=device),
                        Circle(center=torch.tensor([0.0,2.45]),radius=0.3,device=device),
                        ]

    # # plot the figure in the paper
    plot_fig1(scene_1)

    # # plot gradient projection
    # plot_projection(scene_2)