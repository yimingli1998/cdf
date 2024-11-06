# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import pybullet as p
import pybullet_data as pd
import numpy as np
import sys
import time
from pybullet_panda_sim import PandaSim, SphereManager
import torch
import os
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
from nn_cdf import CDF

def main_loop():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    cdf = CDF(device)

    # cdf.np_csdf(torch.rand(100,3).to(device) - torch.tensor([[0.5,0.5,0.0]]).to(device),torch.rand(10,7).to(device),visualize=False)

    # trainer.train_nn(epoches=20000)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
    model.to(device)


    # p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
    #                          ' --background_color_blue=0.5 --width=1600 --height=1000')
    p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                             ' --background_color_blue=1 --width=1000 --height=1000')

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(lightPosition=[5, 5, 5])
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=145, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.6])

    p.setAdditionalSearchPath(pd.getDataPath())
    timeStep = 0.01
    p.setTimeStep(timeStep)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(1)
    ## spawn franka robot
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    panda = PandaSim(p, base_pos, base_rot)
    q0 = panda.get_joint_positions()
    sphere_manager = SphereManager(p)
    # sphere_center = [0.3, 0.4, 0.5]
    # sphere_manager.create_sphere(sphere_center, 0.05, [0.8500, 0.3250, 0.0980, 1])
    q_init = torch.tensor([q0],requires_grad=True).to(device).float()
    while True:
        t0 = time.time()
        sphere_center = np.random.rand(3)*0.5
        # sphere_center[2] += 0.5
        # print(f'sphere_center: {sphere_center}')
        x = torch.tensor([sphere_center],requires_grad=True).to(device).float()
        # print(f'x: {x.shape}, q: {q.shape}')
        q = q_init
        d,grad = cdf.inference_d_wrt_q(x,q,model)
        # print(f'd: {d}, grad: {grad}')
        q_proj = cdf.projection(q,d,grad)
        q_init = q_proj
        print(f'x: {x}')
        print(f'q_proj: {q_proj}')
        panda.set_joint_positions(q_proj[0].data.cpu().numpy())
        # q0[0] = (q0[0]+0.1)%np.pi
        p.addUserDebugText(f"FPS:{int(1/(time.time()-t0))}", [-0.6,0.0,0.8], textColorRGB=(1, 0, 0), textSize=2.0)
        sphere_manager.create_sphere(sphere_center, 0.05, [0.8500, 0.3250, 0.0980, 1])
        time.sleep(1.0)
        p.removeAllUserDebugItems()
        sphere_manager.delete_spheres()

if __name__ == '__main__':
    main_loop()