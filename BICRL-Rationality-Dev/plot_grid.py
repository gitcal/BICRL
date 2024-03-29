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

    # ax.plot(0,0,'x',color='green', markersize=15, markeredgewidth=5)




    data = np.ones((Ny, Nx)) 
    
        # data[cnstr[0], cnstr[1]] = 10

    # data = np.zeros((Ny, Nx)) 
    if trajectories != None:
        for i in trajectories:           
            cnstr = state_grid_map[i[0]]
            data[Ny - 1 - cnstr[1], cnstr[0]] += 1
            # data[cnstr[0], cnstr[1]] = 10
    for ind in np.nonzero(constraints)[0]:
        cnstr = state_grid_map[ind]
        data[Ny - 1 - cnstr[1], cnstr[0]] = 0

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # im1 = plt.imshow(data2, cmap=plt.cm.viridis)
    cmap=plt.cm.viridis#mpl.cm.get_cmap("OrRd").copy()
    cmap.set_under('r')


    # cmap = colors.ListedColormap(['white', 'red'])
    bounds = [0, 10, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.1, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.grid()
    # plt.savefig('plot_act/MAP_results_' + str(kk) + '.png')

    plt.show()

    # return ax




def plot_grid_mean_constr(Nx, Ny, state_grid_map, kk, mean_constraints=None, trajectories=None,  optimal_policy=None):
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



    data = np.zeros((Ny, Nx)) 
    
        # data[cnstr[0], cnstr[1]] = 10


       
    for ind in np.nonzero(mean_constraints)[0]:
        # IPython.embed()
        cnstr = state_grid_map[ind]
        data[Ny - 1 - cnstr[1], cnstr[0]] = mean_constraints[ind]

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    
    cmap = LinearSegmentedColormap.from_list('rg',["w", "r"], N=256)
    # cmap = plt.cm.get_cmap("Reds")
    im = plt.imshow(data, vmin=0.0, vmax=1, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    ax.grid(False)
    plt.savefig('plot_act/mean_results_Scobee_comp'+ str(kk) + '.png')
    # plt.show()
    # plt.pause(3)
    # plt.close()



def plot_performance(list_tupls):
    sns.set(style="whitegrid")
    TPR = []
    FPR = []
    FNR = []
    
    for tup in list_tupls:
        TPR.append(tup[1])
        FPR.append(tup[2])
        FNR.append(tup[3])
    # IPython.embed()
    
    plt.clf()   

    index = pd.RangeIndex(start=0, stop=len(list(range(tup[0])))+1, step=None, name="Number of Iterations")
    data = np.concatenate((np.expand_dims(np.array(TPR),1), np.expand_dims(np.array(FPR),1),np.expand_dims(np.array(FNR),1)), axis=1)
    palette = {"TPR":"tab:green",
           "FPR":"tab:orange",
           "FNR":"tab:blue"}
    markers = {"TPR": "o", "FPR": "o", "FNR": "o"}
    wide_df = pd.DataFrame(data, index, ["TPR", "FPR", "FNR"])
    sns.scatterplot(data=wide_df, palette=palette, markers=markers,alpha=0.6,edgecolor=None)
  
     
    # fig = plt.plot()
    # plt.scatter(list(range(tup[0]+1)),TPR, color='green', label='TPR')
    # plt.scatter(list(range(tup[0]+1)),FPR, color='red', label='FPR')
    # IPython.embed()
    plt.yticks(np.arange(0, 1.01, 0.25))
    plt.xticks(np.arange(0, len(list(range(tup[0])))+2, int((len(list(range(tup[0])))+1)/4)))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Rates')
    plt.legend()
    plt.savefig('plots/perf.png')
    plt.savefig('plots/perf.eps', format='eps')

