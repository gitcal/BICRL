import numpy as np
import IPython
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
from matplotlib.patches import Wedge
import plot_grid
from scipy.interpolate import make_interp_spline





data_traj = []
for i in range(2,22):
    temp = np.loadtxt('data/t'+str(i)+'.txt', delimiter=',')
    data_traj.append(temp)




fig, ax = plt.subplots()
r_const = 0.05
r_constr = 0.1
r_goal = 0.1
constr=[0.5,0.5]
start=[0.5,0]
goal=[0.5,1]

circle1 = plt.Circle((constr[0], constr[1]), r_constr, color='r', alpha=0.5)
circle2 = plt.Circle((start[0], start[1]), r_const, color='blue', alpha=0.5)

plt.text(0.48, 0.02, r'$s_s$',fontsize=36, weight='bold')
plt.text(0.47, 0.92, r'$S_g$',fontsize=36, weight='bold')
plt.text(0.47, 0.48, r'$S_c$',fontsize=36, weight='bold')
ax.add_patch(circle1)
ax.add_patch(circle2)
theta1, theta2 = -180, 2*180
wedge = Wedge(goal, r_goal, theta1, theta2, color='green', alpha=0.5)#, fc=colors[0])#, **kwargs)
ax.add_artist(wedge)



 
for i in range(len(data_traj)):
    data_temp = np.array(data_traj[i])
    list1, list2 = zip(*sorted(zip(data_temp[:,1], data_temp[:,0])))
    X_Y_Spline = make_interp_spline(list1, list2)
    list1 = np.array(list1)
    list2 = np.array(list2)
    X_ = np.linspace(list1.min(), list1.max(), 30)
  

    if i==30:
        plt.plot(data_temp[:,0], data_temp[:,1],'tab:gray', linewidth=3, alpha=0.7)
    else:
        plt.plot(data_temp[:,0], data_temp[:,1],'tab:gray', linewidth=3, alpha=0.7)
    



ax.tick_params(axis = 'both', which = 'major', labelsize = 36)
ax.tick_params(axis = 'both', which = 'minor', labelsize = 36)


ax.axes.xaxis.set_ticks(list(np.arange(0,1.01,1)))
ax.axes.yaxis.set_ticks(list(np.arange(0,1.01,1)))
plt.xlim([0, 1.0])
plt.ylim([0, 1.0])
plt.xlabel("x", fontsize=28)
plt.ylabel("y", fontsize=28)


ax.grid(False)
# plt.savefig('plots/MAP' + str(kk) + '.png')

plt.show()





def plot_cont_2D_v2(env, trajectories, start=[0.5,0], goal=[0.5,1], constr=[0.5,0.5], r_constr = 0.15, r_goal = 0.1):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """

    
    fig, ax = plt.subplots()
    r_const = 0.02
    
  
    circle1 = plt.Circle((constr[0], constr[1]), r_constr, color='r', alpha=0.5)
    circle2 = plt.Circle((start[0], start[1]), r_const, color='blue', alpha=0.5)

    plt.text(0.485, 0.03, r'$s_s$')
    plt.text(0.485, 0.95, r'$S_g$')
    plt.text(0.485, 0.5, r'$S_c$')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    theta1, theta2 = -180, 2*180
    wedge = Wedge(goal, r_goal, theta1, theta2, color='green', alpha=0.5)#, fc=colors[0])#, **kwargs)
    ax.add_artist(wedge)
    ax.axes.xaxis.set_ticks(list(np.arange(0,1.01,1)))
    ax.axes.yaxis.set_ticks(list(np.arange(0,1.01,1)))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()




def plot_grid(Nx, Ny, state_grid_map, kk, constraints=None, trajectories=None,  optimal_policy=None):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    act_dict = {}
    dx_dy = 0.1
    act_dict[0] = (0, -dx_dy)
    act_dict[1] = (0, dx_dy)
    act_dict[2] = (-dx_dy, 0)
    act_dict[3] = (dx_dy, 0)

    
    fig, ax = plt.subplots()
    
    for i in range(Ny + 1):
        ax.plot(np.arange(Nx + 1) - 0.5, np.ones(Nx + 1) * i - 0.5, color='k')
    for i in range(Nx + 1):
        ax.plot(np.ones(Ny + 1) * i - 0.5, np.arange(Ny + 1) - 0.5, color='k')

    y_grid = np.arange(Ny)
    x_grid = np.arange(Nx)
    if optimal_policy != None:
        cnt = 0
        for y in y_grid:
            for x in x_grid:
                if y==0 and x==0:
                    cnt += 1
                    continue
                else:
                    act = optimal_policy[cnt]
                    tup = act_dict[act]
                    ax.arrow(x - tup[0], y - tup[1], tup[0], tup[1], head_width=0.15, head_length=0.2, fc='k', ec='k')
                    cnt += 1





    data = np.ones((Ny, Nx)) 
    
    if trajectories != None:
        for i in trajectories:
           
            cnstr = state_grid_map[i[0]]
            data[Ny - 1 - cnstr[1], cnstr[0]] += 1
            # data[cnstr[0], cnstr[1]] = 10
    for ind in np.nonzero(constraints)[0]:
        # IPython.embed()
        cnstr = state_grid_map[ind]

        data[Ny - 1 - int(round((Ny-1)*cnstr[1],1)), int(round((Ny-1)*cnstr[0],1))] = 0

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
   
    cmap = cmap=plt.cm.viridis#mpl.cm.get_cmap("OrRd").copy()
    cmap.set_under('r')


    # cmap = colors.ListedColormap(['white', 'red'])
    bounds = [0, 10, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.1, cmap=cmap)


    ax.grid(False)
    # plt.savefig('plots/MAP' + str(kk) + '.png')

    plt.show()

