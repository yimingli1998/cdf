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
    
    # trainer.train_nn(epoches=20000)
    model = MLPRegression(input_dims=10, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
    model.to(device)

    p.connect(p.GUI, options='--background_color_red=0.5 --background_color_green=0.5' +
                                ' --background_color_blue=0.5 --width=1600 --height=1000')
    

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(lightPosition=[5, 5, 5])
    p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000)
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-10, cameraTargetPosition=[0, 0, 0.5])
    # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=90, cameraPitch=0, cameraTargetPosition=[0, 0, 0.5])
    #p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=110, cameraPitch=-25, cameraTargetPosition=[0, 0, 0.5])
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=145, cameraPitch=0, cameraTargetPosition=[0, 0, 0.6])
    p.setAdditionalSearchPath(pd.getDataPath())
        
    ## spawn franka robot
    base_pos = [0, 0, 0]
    base_rot = p.getQuaternionFromEuler([0, 0, 0])
    robot = PandaSim(p, base_pos, base_rot)
    q0 = robot.get_joint_positions()
    q_init = torch.tensor([q0],requires_grad=True).to(device).float()
    # NOTE: need high frequency
    hz = 1000
    delta_t = 1.0 / hz
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(delta_t)
    p.setRealTimeSimulation(0)


    p.setAdditionalSearchPath(pd.getDataPath())
    soccerballId = p.loadURDF("soccerball.urdf", [-3.0, 0, 3], globalScaling=0.1)
    p.changeDynamics(soccerballId, -1, mass=1.0, linearDamping=0.00, angularDamping=0.00, rollingFriction=0.03,
                    spinningFriction=0.03, restitution=0.2, lateralFriction=0.03)
    
    box_size = np.array([0.6,0.01,0.3])
    box_center = np.array([0.0,0.3,0.3])
    box = p.createVisualShape(p.GEOM_BOX, halfExtents=box_size, rgbaColor=[0.8500, 0.3250, 0.0980, 1.0])
    p.createMultiBody(baseVisualShapeIndex=box,
                                        basePosition=box_center)
    ball_pos = np.array([0.0, -1.0, 0])
    vec_max = np.array([1.0, 3.0, 4.0])
    vec_min = np.array([-1.0, 2.0, 3.0])
    ball_vec = np.random.rand(100,3) * (vec_max - vec_min) + vec_min
    for vec in ball_vec:
        for _ in range(300):
            robot.set_joint_positions(q0)
            p.stepSimulation()
        p.resetBasePositionAndOrientation(soccerballId, ball_pos, [0, 0, 0, 1])
        p.resetBaseVelocity(soccerballId, linearVelocity=vec)
        defense,proj = False, False
        while(True):
            # print(robot.get_joint_positions())
            position, orientation = p.getBasePositionAndOrientation(soccerballId)
            if (position[0] > box_center[0]-box_size[0]) & (position[0] < box_center[0] + box_size[0]) & \
                (position[1] > -0.5) & (position[1] < -0.1) &(position[2] > 0.0) & (position[2] < box_center[2]+box_size[2] + 0.2):
                defense = True
            if defense and not proj:
                print('defense')
                q = q_init
                x = torch.from_numpy(np.array([position])).to(device).float()
                d,grad = cdf.inference_d_wrt_q(x,q,model)
                # print(f'd: {d}, grad: {grad}')
                q_proj = cdf.projection(q,d,grad)
                q_init = q_proj
                robot.set_joint_positions(q_proj[0].data.cpu().numpy())
                proj = True

            p.stepSimulation()
            time.sleep(delta_t*2.0)
            if position[2] < -0.1 or position[2] > 1.5 or position[1] < -1.0:
                break

    p.disconnect()


if __name__ == '__main__':

    main_loop()