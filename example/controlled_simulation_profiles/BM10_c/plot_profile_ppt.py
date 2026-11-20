import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.patches as mpatches
import math
from math import log
from math import sqrt
from scipy.interpolate import interp1d
from scipy.integrate import quad
#from astropy.modeling.physical_models import NFW
#import astropy.constants as c
#import astropy.units as u
#import pynbody as nb
import matplotlib.cm as cm

data10 = np.genfromtxt('./prof_bound_halo_snapshot_snap000.txt', names=True)
data11 = np.genfromtxt('./prof_bound_halo_snapshot_snap010.txt', names=True)
data12 = np.genfromtxt('./prof_bound_halo_snapshot_snap010.txt', names=True)

data20 = np.genfromtxt('./prof_bound_stars_snapshot_snap000.txt', names=True)
data21 = np.genfromtxt('./prof_bound_stars_snapshot_snap010.txt', names=True)
data22 = np.genfromtxt('./prof_bound_stars_snapshot_snap010.txt', names=True)

h = 1

fig = plt.figure(figsize=(6, 6))
plt.xlabel(r"$r\ ({\rm kpc})$",fontsize=12)
plt.ylabel(r"$\rho \ (M_{\odot} {\rm kpc}^{-3}) $",fontsize=12)

cmap = cm.get_cmap('coolwarm')
colorlist = cmap(np.linspace(0, 1, 10))

plt.plot(data10['r']/h, (data10['rho'])*h*h, linewidth=2, color=colorlist[0], label=r'0 Gyr (DM)', alpha=0.7)
plt.plot(data12['r']/h, (data12['rho'])*h*h, linewidth=2, color=colorlist[8], label=r'2 Gyr', alpha=0.7)

plt.plot(data20['r']/h, data20['rho']*h*h, '--', linewidth=2, color=colorlist[0], label=r'0 Gyr (Star)', alpha=0.7)
plt.plot(data22['r']/h, data22['rho']*h*h, '--', linewidth=2, color=colorlist[8], label=r'2 yr', alpha=0.7)
"""
plt.plot(data30['r']/h, data30['rho']*h*h, '-.', linewidth=2, color=colorlist[0], label=r'0 Gyr (stars)', alpha=0.7)
plt.plot(data32['r']/h, data32['rho']*h*h, '-.', linewidth=2, color=colorlist[8], label=r'5 Gyr', alpha=0.7)


plt.plot(data10['r']/h, (data10['rho']+data20['rho']+data30['rho'])*h*h,'-.' ,linewidth=4, color=colorlist[0], label=r'0 Gyr (tot)', alpha=0.7)
plt.plot(data11['r']/h, (data11['rho']+data21['rho']+data31['rho'])*h*h, '-.' ,linewidth=4, color=colorlist[5], label=r'1 Gyr', alpha=0.7)
plt.plot(data12['r']/h, (data12['rho']+data22['rho']+data32['rho'])*h*h,'-.' , linewidth=4, color=colorlist[8], label=r'2 Gyr', alpha=0.7)
"""

plt.ylim([1e6,3e9])
plt.xlim([0.2, 10])

plt.xscale('log')
plt.yscale('log')

plt.legend(ncol=2)
plt.show()
fig.savefig('NH2_sidm_profile.pdf', bbox_inches='tight')

