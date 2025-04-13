# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------

import torch
import os
from cdf import CDF2D
import nn_cdf 
import casadi as cs
import l4casadi as l4c

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cdf = nn_cdf.Train_CDF(device)
    
    net = torch.load(os.path.join('./2Dexamples', 'model.pth'))
    net.eval()


    l4c_model = l4c.L4CasADi(net, device='cuda')  # device='cuda' for GPU
    x_sym = cs.MX.sym('x', 1, 4)
    y_sym = l4c_model(x_sym)
    f = cs.Function('y', [x_sym], [y_sym])
    df = cs.Function('dy', [x_sym], [cs.jacobian(y_sym, x_sym)])
    ddf = cs.Function('ddy', [x_sym], [cs.hessian(y_sym, x_sym)[0]])

    x = cs.DM([[0.], [2.],[0.], [2.]])
    print(l4c_model(x))
    print(f(x))
    print(df(x))
    print(ddf(x))