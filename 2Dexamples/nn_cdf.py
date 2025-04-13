# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
from mlp import MLPRegression
import os
from cdf import CDF2D
from tqdm import tqdm


class Train_CDF:
    def __init__(self,device) -> None:
        self.device = device  

        self.cdf = CDF2D(device)

        self.q_template = self.cdf.q_grid_template.view(-1,200,2)

        x = torch.linspace(self.cdf.task_space[0][0],self.cdf.task_space[1][0],self.cdf.nbData).to(self.device)
        y = torch.linspace(self.cdf.task_space[0][1],self.cdf.task_space[1][1],self.cdf.nbData).to(self.device)
        xx,yy = torch.meshgrid(x,y)
        xx,yy = xx.reshape(-1,1),yy.reshape(-1,1)
        self.p = torch.cat([xx,yy],dim=-1).to(self.device)


    def matching_csdf(self,q):
        # q: [batchsize,2]
        # return d:[len(x),len(q)]
        dist = torch.norm(q.unsqueeze(1).expand(-1,200,-1) - self.q_template.unsqueeze(1),dim=-1)
        d,idx = torch.min(dist,dim=-1)
        q_template = torch.gather(self.q_template,1,idx.unsqueeze(-1).expand(-1,-1,2))
        return d,q_template


    def train(self,input_dim, hidden_dim, output_dim, activate, batch_size, learning_rate, weight_decay, save_path, device,
          epochs):
        # model
        net = MLPRegression(input_dims=input_dim,
                            output_dims=output_dim, 
                            mlp_layers=hidden_dim,
                            skips=[],
                            act_fn=activate, 
                            nerf=True).to(device)
        # net.apply(model.init_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        batch_p = self.p.unsqueeze(1).expand(-1,batch_size,-1).reshape(-1,2)
        max_loss = float("inf")
        net.train()
        for i in tqdm(range(epochs)):
            q = torch.rand(batch_size,2,requires_grad=True).to(self.device)*2*torch.pi-torch.pi
            batch_q = q.unsqueeze(0).expand(len(self.p),-1,-1).reshape(-1,2)

            d,q_temp = self.matching_csdf(q)
            q_temp = q_temp.reshape(-1,2)
            mask = d.reshape(-1)<torch.inf
            # mask = d<torch.inf
            # print(d.shape,_p.shape,q.shape)
            inputs = torch.cat([batch_p,batch_q],dim=-1).reshape(-1,4)
            outputs = d.reshape(-1,1)
            inputs,outputs = inputs[mask],outputs[mask]
            q_temp = q_temp[mask]
            weights = torch.ones_like(outputs).to(device)
            # weights = (1/outputs).clamp(0,1)

            d_pred = net.forward(inputs)
            d_grad_pred = torch.autograd.grad(d_pred, batch_q, torch.ones_like(d_pred), retain_graph=True)[0]
            d_grad_pred = d_grad_pred[mask]

            # Compute the Eikonal loss
            eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

            # Compute the MSE loss
            d_loss = ((d_pred-outputs)**2*weights).mean()

            # Compute the projection loss
            proj_q = batch_q[mask] - d_grad_pred*d_pred
            proj_loss = torch.norm(proj_q-q_temp,dim=-1).mean()

            # Combine the two losses with appropriate weights
            w0 = 1.0
            w1 = 1.0
            w2 = 0.1
            loss = w0 * d_loss + w1 * eikonal_loss + w2*proj_loss
            print(f"Epoch {i+1}/{epochs}, Loss: {loss.item():.4f}, d_loss: {d_loss.item():.4f}, eikonal_loss: {eikonal_loss.item():.4f}, proj_loss: {proj_loss.item():.4f}")
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if loss.item() < max_loss:
                max_loss = loss.item()
                torch.save(net, os.path.join(save_path, 'model.pth'))

def inference(x,q,net):
    x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,2)
    q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,2)
    inputs = torch.cat([x_cat,q_cat],dim=-1)
    c_dist = net.forward(inputs).squeeze()
    grad = torch.autograd.grad(c_dist, q_cat, torch.ones_like(c_dist), retain_graph=True)[0]
    # print(c_dist.shape,grad.shape)
    return c_dist.squeeze(),grad


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cdf = Train_CDF(device)
    # train_cdf.train(input_dim=4, 
    #           hidden_dim=[256, 256, 128, 128, 128], 
    #           output_dim=1, 
    #           activate=torch.nn.ReLU, 
    #           batch_size=100,
    #           learning_rate=0.01, 
    #           weight_decay=1e-5, 
    #           save_path='./2Dexamples', 
    #           device=device,
    #       epochs=1000)
    
    net = torch.load(os.path.join('./2Dexamples', 'model.pth'))
    net.eval()
    x = torch.tensor([[2.0,2.0]],device=device)
    q = train_cdf.cdf.create_grid_torch(train_cdf.cdf.nbData).to(device)
    q.requires_grad = True
    q_proj = q.clone()
    for i in range (10):
        c_dist,grad = inference(x,q_proj,net)
        q_proj = train_cdf.cdf.projection(q_proj,c_dist,grad)
        # print(q_proj.shape)
    # plot
    import matplotlib.pyplot as plt
    c_dist,grad = inference(x,q,net)
    plt.contourf(q[:,0].detach().cpu().numpy().reshape(50,50), q[:,1].detach().cpu().numpy().reshape(50,50), c_dist.cpu().detach().numpy().reshape(50,50), levels=20)
    plt.scatter(q_proj[:,0].detach().cpu().numpy(), q_proj[:,1].detach().cpu().numpy(), c='r', s=1)
    plt.title('CDF')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # to learn