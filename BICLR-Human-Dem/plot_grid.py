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

def plot_cont_2D(traj, goal=[0.5,0.5]):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """    
    fig, ax = plt.subplots()   
  

    ax.plot(traj[0][0][0],traj[0][0][1],'x',color='green', markersize=15, markeredgewidth=5)
    ax.plot(goal[0],goal[1],'x',color='red', markersize=15, markeredgewidth=5)
    x_traj = []
    y_traj = []
    for demo in traj:
        # IPython.embed()
        x_traj.append(demo[0][0])
        y_traj.append(demo[0][1])
        # ax.plot(traj[0][0],traj[0][1])
    # IPython.embed()
    ax.plot(x_traj,y_traj)
    ax.axes.xaxis.set_ticks(list(np.arange(0,1.01,0.1)))
    ax.axes.yaxis.set_ticks(list(np.arange(0,1.01,0.1)))


    data_traj = []
    for i in range(9,31):
        temp = np.loadtxt('t'+str(i)+'.txt', delimiter=',')
        data_traj.append(temp)
   
    
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
    
 
    
    for traj in trajectories:
        new_list = []
        for i in traj:           
            new_list.append((env.num_state_grid_map[i[0]]))
        # IPython.embed()
        new_list = np.array(new_list)
        plt.plot(new_list[:,0],new_list[:,1],color='blue', alpha=0.6)
   
    circle1 = plt.Circle((constr[0], constr[1]), r_constr, color='r', alpha=0.5)
    circle2 = plt.Circle((start[0], start[1]), r_const, color='blue', alpha=0.5)

    plt.text(0.485, -0.06, r'$s_s$')
    plt.text(0.485, 0.95, r'$S_g$')
    plt.text(0.485, 0.5, r'$S_c$')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    theta1, theta2 = -180, 2*180
    wedge = Wedge(goal, r_goal, theta1, theta2, color='green', alpha=0.5)#, fc=colors[0])#, **kwargs)
    ax.add_artist(wedge)
    ax.axes.xaxis.set_ticks(list(np.arange(0,1.01,0.1)))
    ax.axes.yaxis.set_ticks(list(np.arange(0,1.01,0.1)))
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def plot_grid_test(Nx, Ny, state_grid_map, V):
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
        # IPython.embed()

    y_grid = np.arange(Ny)
    x_grid = np.arange(Nx)
   

    ax.plot(0.25,0.25,'x',color='green', markersize=15, markeredgewidth=5)
    ax.plot(0.9,0.9,'x',color='red', markersize=15, markeredgewidth=5)



    # data = np.ones((Ny, Nx)) 
    
        # data[cnstr[0], cnstr[1]] = 10

    data = np.zeros((Ny+1, Nx+1)) 

    for i in range(len(V)):
        coords = state_grid_map[i]
        # IPython.embed()
        data[10-int(10*coords[0]),int(10*coords[1])]=V[i]


    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    
    # im1 = plt.imshow(data2, cmap=plt.cm.viridis)
    cmap = cmap=plt.cm.viridis#mpl.cm.get_cmap("OrRd").copy()
    cmap.set_under('r')


    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.1, cmap=cmap)
    
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.grid(False)
    # plt.savefig('plots/MAP' + str(kk) + '.png')
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
        # IPython.embed()

    y_grid = np.arange(Ny)
    x_grid = np.arange(Nx)
    # IPython.embed()
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
            
    for ind in np.nonzero(constraints)[0]:
        
        cnstr = state_grid_map[ind]

        data[Ny - 1 - int(round((Ny-1)*cnstr[1],1)), int(round((Ny-1)*cnstr[0],1))] = 0

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    
    
    # im1 = plt.imshow(data2, cmap=plt.cm.viridis)
    cmap = cmap=plt.cm.viridis_r#mpl.cm.get_cmap("OrRd").copy()
    cmap.set_under('r')


    # cmap = colors.ListedColormap(['white', 'red'])
    bounds = [0, 10, 20]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.1, cmap=cmap)
    
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    ax.grid(False)
    # plt.savefig('plots/MAP' + str(kk) + '.png')

    plt.show()

    # return ax

def plot_posterior(Nx, Ny, posterior, ii, constraints, unc_flag=False, state_grid_map=None):
    """
    Plots the skeleton of the grid world
    :param ax:
    :return:
    """
    if not unc_flag:
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        cnt = 1
        for i in range(Ny):
            for j in range(Nx):
                ax = fig.add_subplot(Ny, Nx, cnt)
                if cnt-1 in constraints:
                    ax.hist(posterior[cnt-1], color = 'red')
                else:
                    ax.hist(posterior[cnt-1])
                plt.xlim([-0.5, 1.5])
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(4) 
                plt.yticks([], [])
                cnt += 1
    else:

        fig, ax = plt.subplots()
        
        for i in range(Ny + 1):
            ax.plot(np.arange(Nx + 1) - 0.5, np.ones(Nx + 1) * i - 0.5, color='k')
        for i in range(Nx + 1):
            ax.plot(np.ones(Ny + 1) * i - 0.5, np.arange(Ny + 1) - 0.5, color='k')


        ax.plot(0,0,'x',color='green', markersize=15, markeredgewidth=5)

        data = np.zeros((Ny, Nx)) 
       
        for ind in range(Ny * Nx):
           
            cnstr = state_grid_map[ind]
            
            theta = posterior[ind]
            data[Ny - 1 - cnstr[1], cnstr[0]] = theta * (1-theta)


        cmap=plt.cm.viridis # mpl.cm.get_cmap("OrRd").copy()
       
        im = plt.imshow(data, vmin=0.0, vmax=0.25, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        plt.colorbar(im, cax=cax)
       
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # plt.colorbar()
        ax.grid(False)

        # plt.savefig('plots/posterior_var'+ str(ii) +'.png')
        plt.show()



def plot_grid_mean_constr(Nx, Ny, state_grid_map, kk=0, constraints_plot=None, mean_constraints=None, trajectories=None,  optimal_policy=None):
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
    
    for ind in np.nonzero(mean_constraints)[0]:
        cnstr = state_grid_map[ind]
        data[Ny - 1 - int(round((Ny-1)*cnstr[1],1)), int(round((Ny-1)*cnstr[0],1))] = mean_constraints[ind]

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    
    
    cmap = LinearSegmentedColormap.from_list('rg',["w", "r"], N=256)
    # cmap = plt.cm.get_cmap("Reds")
    # cmap.set_under('r')
    # cmap = colors.ListedColormap(['white', 'red'])
    # bounds = [0, 10, 20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(data, vmin=0.0, vmax=1, cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # plt.colorbar()
    ax.grid(False)
    plt.savefig('plots/fourth_mean_results' + str(kk) + '.png')
    # plt.show(block=False)
    # plt.pause(5)
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
    plt.yticks(np.arange(0, 1.01, 0.25))
    plt.xticks(np.arange(0, len(list(range(tup[0])))+2, int((len(list(range(tup[0])))+1)/4)))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Rates')
    plt.legend()
    plt.savefig('plots/perf.png')
    plt.savefig('plots/perf.eps', format='eps')

