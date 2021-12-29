import numpy as np
import IPython
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats 

data = np.loadtxt('penalty_rew0.txt')

mean, var  = scipy.stats.distributions.norm.fit(data[int(np.floor(len(data)*0.4)):])

x = np.linspace(min(data),max(data)+1,100)

fitted_data = scipy.stats.distributions.norm.pdf(x, mean, var)
plt.hist(data[int(np.floor(len(data)*0.4)):], density=True, bins = 20)
plt.plot(x,fitted_data,'r-')


IPython.embed()
plt.xlabel('Penalty reward')
plt.ylabel('Frequency')
plt.xlim([-16, -8])
plt.grid(False)
plt.savefig('plots/post_rew_distr.png')
plt.show()


# plt.show()


