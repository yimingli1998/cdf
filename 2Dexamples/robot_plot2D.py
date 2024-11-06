# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the CDF project.
# Copyright (c) 2024 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


"""
Fancy visualization for a planar manipulator

Copyright (c) 2023 Idiap Research Institute <https://www.idiap.ch/>
Written by Boyang Ti

This file is part of RCFS <https://robotics-codes-from-scratch.github.io/>
License: MIT
"""
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
import matplotlib.cm as cm

def plotArmLink(ax, a, d, p, sz, facecol, edgecol, **kwargs):
    nbSegm = 30

    Path = mpath.Path

    # calculate the link border
    xTmp = np.zeros((2, nbSegm))
    p = p + np.array([0, 0]).reshape(2, -1)
    t1 = np.linspace(0, -np.pi, int(nbSegm / 2))
    t2 = np.linspace(np.pi, 0, int(nbSegm / 2))
    xTmp[0, :] = np.hstack((sz * np.sin(t1), d + sz * np.sin(t2)))
    xTmp[1, :] = np.hstack((sz * np.cos(t1), sz * np.cos(t2)))
    # xTmp[2, :] = np.zeros((1, nbSegm))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    x = R @ xTmp + np.matlib.repmat(p, 1, nbSegm)
    p2 = R @ np.array([d, 0]).reshape(2, -1) + p

    # add the link patch
    codes = Path.LINETO * np.ones(np.size(x[0:2, :], 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path(x[0:2, :].T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    # add the initial point
    msh = (
        np.vstack(
            (
                np.sin(np.linspace(0, 2 * np.pi, nbSegm)),
                np.cos(np.linspace(0, 2 * np.pi, nbSegm)),
            )
        )
        * sz
        * 0.4
    )

    codes = Path.LINETO * np.ones(np.size(msh[0:2, :], 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path((msh[0:2, :] + p).T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)
    # add the end point
    path = Path((msh[0:2, :] + p2).T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    return p2


def plotArmBasis(ax, p1, sz, facecol, edgecol, **kwargs):
    Path = mpath.Path

    nbSegm = 30
    sz = sz * 1.2

    xTmp1 = np.zeros((2, nbSegm))
    t1 = np.linspace(0, np.pi, nbSegm - 2)
    xTmp1[0, :] = np.hstack([sz * 1.5, sz * 1.5 * np.cos(t1), -sz * 1.5])
    xTmp1[1, :] = np.hstack([-sz * 1.2, sz * 1.5 * np.sin(t1), -sz * 1.2])
    x1 = xTmp1 + np.matlib.repmat(p1, 1, nbSegm)
    # add the link patch
    codes = Path.LINETO * np.ones(np.size(x1, 1), dtype=Path.code_type)
    codes[0] = Path.MOVETO
    path = Path(x1.T, codes)
    patch = PathPatch(path, facecolor=facecol, edgecolor=edgecol, **kwargs)
    ax.add_patch(patch)

    nb_line = 4
    mult = 1.2
    xTmp2 = np.zeros((2, nb_line))  # 2D only
    xTmp2[0, :] = np.linspace(-sz * mult, sz * mult, nb_line)
    xTmp2[1, :] = [-sz * mult] * nb_line

    x2 = xTmp2 + np.tile((p1.flatten() + np.array([0.0, 0.0]))[:, None], (1, nb_line))
    x3 = xTmp2 + np.tile(
        (p1.flatten() + np.array([-0.2, -0.8]) * sz)[:, None], (1, nb_line)
    )

    for i in range(nb_line):
        tmp = np.zeros((2, 2))  # N*2
        tmp[0] = [x2[0, i], x2[1, i]]
        tmp[1] = [x3[0, i], x3[1, i]]
        patch = Line2D(tmp[:, 0], tmp[:, 1], color=[0, 0, 0, 1], lw=2, zorder=1)
        ax.add_line(patch)


def plotArm(
    ax,
    a,
    d,
    p,
    sz=0.1,
    facecol=None,
    edgecol=None,
    xlim=None,
    ylim=None,
    robot_base=False,
    **kwargs
):
    if edgecol is None:
        edgecol = [0.99, 0.99, 0.99]
    if facecol is None:
        facecol = [0.5, 0.5, 0.5]
    p = np.reshape(p, (-1, 1))
    if robot_base:
        plotArmBasis(ax, p, sz, facecol, edgecol, **kwargs)
    for i in range(len(a)):
        p = plotArmLink(
            ax=ax,
            a=np.sum(a[0 : i + 1]),
            d=d[i],
            p=p + np.array([0.0, 0.0]).reshape(2, -1),
            sz=sz,
            facecol=facecol,
            edgecol=edgecol,
            **kwargs
        )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is None and ylim is None:
        ax.autoscale(True)


def plotArm_Tool(
    ax,
    a,
    d,
    p,
    sz=0.1,
    facecol=None,
    edgecol=None,
    xlim=None,
    ylim=None,
    robot_base=False,
    **kwargs
):
    if edgecol is None:
        edgecol = [0.99, 0.99, 0.99]
    if facecol is None:
        facecol = [0.5, 0.5, 0.5]
    p = np.reshape(p, (-1, 1))
    if robot_base:
        plotArmBasis(ax, p, sz, facecol, edgecol, **kwargs)
    for i in range(len(a)):
        if i == len(a) - 1:
            p = plotArmLink(
                ax=ax,
                a=np.sum(a[0 : i + 1]),
                d=d[i],
                p=p + np.array([0.0, 0.0]).reshape(2, -1),
                sz=sz,
                facecol=facecol,
                edgecol=edgecol,
                alpha=0.4,
            )
        else:
            p = plotArmLink(
                ax=ax,
                a=np.sum(a[0 : i + 1]),
                d=d[i],
                p=p + np.array([0.0, 0.0]).reshape(2, -1),
                sz=sz,
                facecol=facecol,
                edgecol=edgecol,
                **kwargs
            )
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is None and ylim is None:
        ax.autoscale(True)


def plot_planar_axis(ax, p):
    length = 0.2
    num = np.size(p, 0)
    for i in range(num):
        x_1 = np.array([p[i, 0], p[i, 0] + length * np.cos(p[i, 2])])
        y_1 = np.array([p[i, 1], p[i, 1] + length * np.sin(p[i, 2])])
        (ln1,) = ax.plot(x_1, y_1, lw=2, solid_capstyle="round", color="r", zorder=1)
        ln1.set_solid_capstyle("round")

        x_2 = np.array([p[i, 0], p[i, 0] + length * np.cos(p[i, 2] + np.pi / 2)])
        y_2 = np.array([p[i, 1], p[i, 1] + length * np.sin(p[i, 2] + np.pi / 2)])
        (ln2,) = ax.plot(x_2, y_2, lw=2, solid_capstyle="round", color="b", zorder=1)
        ln2.set_solid_capstyle("round")


def plot_2d_manipulators(link1_length=2, 
                         link2_length=2, 
                         joint_angles_batch=None,
                         ax=None,
                         color = 'green',
                         alpha = 1.0,
                         show_start_end = False,
                         show_eef_traj = False):
    # Check if joint_angles_batch is None or has incorrect shape
    if joint_angles_batch is None or joint_angles_batch.shape[1] != 2:
        raise ValueError("joint_angles_batch must be provided with shape (N, 2)")

    if ax is None:
        fig, ax = plt.subplots()

    # Number of sets of joint angles
    num_sets = joint_angles_batch.shape[0]


    # Create a figure
    cmap = cm.get_cmap('Reds', num_sets)  # You can choose other colormaps like 'Greens', 'Reds', etc.
    cmap2 = cm.get_cmap('Reds', num_sets)  # You can choose other colormaps like 'Greens', 'Reds', etc.
    # the color will 
    traj_list = []
    for i in range(num_sets):
        if i ==0 or i==num_sets-1:
            alpha = 1.0
            line_width = 3
        else:
            alpha = 0.4
            line_width = 2
        # Extract joint angles for the current set
        theta1, theta2 = joint_angles_batch[i]

        # Calculate the position of the first joint
        joint1_x = link1_length * np.cos(theta1)
        joint1_y = link1_length * np.sin(theta1)

        # Calculate the position of the end effector (tip of the second link)
        end_effector_x = joint1_x + link2_length * np.cos(theta1 + theta2)
        end_effector_y = joint1_y + link2_length * np.sin(theta1 + theta2)

        # Stack the base, joint, and end effector positions
        positions = np.vstack([[0, 0], [joint1_x, joint1_y], [end_effector_x, end_effector_y]])  # shape: (3, 2)
        traj_list.append(positions)
        # Plotting
        if show_start_end:
            if i>0 and i<num_sets-1:
                continue
        
        ax.plot(positions[:, 0], positions[:, 1], linestyle='-', color=color, marker='o', markersize=0, markerfacecolor='white',markeredgecolor='red',alpha= alpha,linewidth=line_width)
        
        # # cover the end effector with different colors to hightlight the trajectory

        # ax.plot(positions[2, 0], positions[2, 1], linestyle='-', color=color, marker='o', markersize=3, markerfacecolor='white',markeredgecolor='red',alpha=alpha,linewidth=2)

        # plot a bigger base center at (0, 0), which is a cirlce with golden color
        ax.plot(0, 0, marker='o', markersize=5, markerfacecolor='grey', markeredgecolor='k')
    traj_list = np.array(traj_list)
    if show_eef_traj:
        ax.plot(traj_list[:,2,0],traj_list[:,2,1],linestyle='--', color='grey')


if __name__ == "__main__":
    fig, ax = plt.subplots()
    x_t = np.array([0.5,1.1])

    from matplotlib import patches
    from args import get_args
    args = get_args()
    for j in range(len(args.center)):
        circle = patches.Circle(args.center[j], args.radius, linewidth=2, edgecolor='black', facecolor='none')                                                                
        plt.gca().add_patch(circle)
    l = [2, 2]  # robot link lengths
    plotArm(
        ax=ax,
        a=x_t,
        d=l,
        p=np.array([0.0, 0.0]),  # base position
        sz=0.08,
        label="via",
        alpha=1.0,
        zorder=2,
        xlim=None,
        ylim=None,
        robot_base=True,  # to visualize the base
    )
    # for i in range(x_t.shape[1]):
    #     if i%20==0:
    #         print(i)
    #         plotArm(
    #             ax=ax,
    #             a=x_t[:,i],
    #             d=l,
    #             p=np.array([0.0, 0.0]),  # base position
    #             sz=0.08,
    #             label="via",
    #             alpha=0.1+i*0.0022,
    #             zorder=2,
    #             xlim=None,
    #             ylim=None,
    #             robot_base=True,  # to visualize the base
    #         )

    ax.set_aspect("equal", "box")
    plt.show()