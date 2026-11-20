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


data10 = np.genfromtxt('prof_bound_halo_snapshot_snap000.txt', names=True)
data11 = np.genfromtxt('prof_bound_halo_snapshot_snap001.txt', names=True)
#data12 = np.genfromtxt('prof_bound_halo_snapshot_snap004.txt', names=True)
data13 = np.genfromtxt('prof_bound_halo_snapshot_snap016.txt', names=True)
data14 = np.genfromtxt('prof_bound_halo_snapshot_snap022.txt', names=True)
data15 = np.genfromtxt('prof_bound_halo_snapshot_snap000.txt', names=True)
data16 = np.genfromtxt('prof_bound_halo_snapshot_snap000.txt', names=True)
data17 = np.genfromtxt('prof_bound_halo_snapshot_snap000.txt', names=True)
"""
data18 = np.genfromtxt('prof_bound_halo_EB5_snap050.txt', names=True)
data19 = np.genfromtxt('prof_bound_halo_EB5_snap060.txt', names=True)
"""
data20 = np.genfromtxt('prof_bound_stars_snapshot_snap000.txt', names=True)
data21 = np.genfromtxt('prof_bound_stars_snapshot_snap001.txt', names=True)
#data22 = np.genfromtxt('prof_bound_stars_snapshot_snap002.txt', names=True)
data23 = np.genfromtxt('prof_bound_stars_snapshot_snap016.txt', names=True)
data24 = np.genfromtxt('prof_bound_stars_snapshot_snap022.txt', names=True)
data25 = np.genfromtxt('prof_bound_stars_snapshot_snap000.txt', names=True)
data26 = np.genfromtxt('prof_bound_stars_snapshot_snap000.txt', names=True)
data27 = np.genfromtxt('prof_bound_stars_snapshot_snap000.txt', names=True)
"""
data28 = np.genfromtxt('prof_bound_stars_EB5_snap050.txt', names=True)
data29 = np.genfromtxt('prof_bound_stars_EB5_snap060.txt', names=True)
"""
h = 1

fig = plt.figure()
plt.xlabel(r"$r\ ({\rm kpc})$",fontsize=12)
plt.ylabel(r"$\rho \ (M_{\odot} {\rm kpc}^{-3}) $",fontsize=12)

cmap = cm.get_cmap('coolwarm')
colorlist = cmap(np.linspace(0, 1, 10))

"""
plt.plot(data10['r']/h, (data10['rho'])*h*h, linewidth=2, color=colorlist[0], label=r'0 Gyr (halo)', alpha=0.7)
plt.plot(data11['r']/h, (data11['rho'])*h*h, linewidth=2, color=colorlist[1], label=r'0.2 Gyr', alpha=0.7)
plt.plot(data12['r']/h, (data12['rho'])*h*h, linewidth=2, color=colorlist[2], label=r'0.4 Gyr', alpha=0.7)
plt.plot(data13['r']/h, (data13['rho'])*h*h, linewidth=2, color=colorlist[3], label=r'2 Gyr', alpha=0.7)
plt.plot(data14['r']/h, (data14['rho'])*h*h, linewidth=2, color=colorlist[4], label=r'4 Gyr', alpha=0.7)
plt.plot(data15['r']/h, (data15['rho'])*h*h, linewidth=2, color=colorlist[5], label=r'6.6 Gyr', alpha=0.7)
plt.plot(data16['r']/h, (data16['rho'])*h*h, linewidth=2, color=colorlist[6], label=r'10 Gyr', alpha=0.7)
plt.plot(data17['r']/h, (data17['rho'])*h*h, linewidth=2, color=colorlist[7], label=r'12 Gyr', alpha=0.7)

plt.plot(data20['r']/h, data20['rho']*h*h, '--', linewidth=2, color=colorlist[0], label=r'0 Gyr (stars)', alpha=0.7)
plt.plot(data21['r']/h, data21['rho']*h*h, '--', linewidth=2, color=colorlist[1], label=r'0.2 Gyr', alpha=0.7)
plt.plot(data22['r']/h, data22['rho']*h*h, '--', linewidth=2, color=colorlist[2], label=r'0.4 yr', alpha=0.7)
plt.plot(data23['r']/h, data23['rho']*h*h, '--', linewidth=2, color=colorlist[3], label=r'2 Gyr', alpha=0.7)
plt.plot(data24['r']/h, data24['rho']*h*h, '--', linewidth=2, color=colorlist[4], label=r'4 Gyr', alpha=0.7)
plt.plot(data25['r']/h, data25['rho']*h*h, '--', linewidth=2, color=colorlist[5], label=r'6.6 Gyr', alpha=0.7)
plt.plot(data26['r']/h, data26['rho']*h*h, '--', linewidth=2, color=colorlist[6], label=r'10 Gyr', alpha=0.7)
plt.plot(data27['r']/h, data27['rho']*h*h, '--', linewidth=2, color=colorlist[7], label=r'12 Gyr', alpha=0.7)
"""
def rhoNFW(r,rhos,rs):
   return rhos/(r/rs)/pow(1+r/rs,2)

plt.plot(data10['r']/h, (data10['rho'])*h*h, linewidth=2, color=colorlist[0], label=r'0 Gyr (halo)', alpha=0.7)
plt.plot(data10['r']/h,rhoNFW(data10['r']/h,20994483.63,35.41),color='g')
plt.plot(data11['r']/h, (data11['rho'])*h*h, linewidth=2, color=colorlist[1], label=r'0.05 Gyr', alpha=0.7)
#plt.plot(data12['r']/h, (data12['rho'])*h*h, linewidth=2, color=colorlist[3], label=r'0.4 Gyr', alpha=0.7)
plt.plot(data13['r']/h, (data13['rho'])*h*h, linewidth=2, color=colorlist[5], label=r'0.1 Gyr', alpha=0.7)
plt.plot(data14['r']/h, (data14['rho'])*h*h, linewidth=2, color=colorlist[8], label=r'0.15 Gyr', alpha=0.7)
#plt.plot(data15['r']/h, (data15['rho'])*h*h, linewidth=2, color=colorlist[8], label=r'$6.6 Gyr (0.2t_{\rm cb})$', alpha=0.7)
#plt.plot(data16['r']/h, (data16['rho'])*h*h, linewidth=2, color=colorlist[8], label=r'10 Gyr', alpha=0.7)
#plt.plot(data17['r']/h, (data17['rho'])*h*h, linewidth=2, color=colorlist[9], label=r'12 Gyr', alpha=0.7)

plt.plot(data20['r']/h, data20['rho']*h*h, '--', linewidth=2, color=colorlist[0], label=r'0 Gyr (stars)', alpha=0.7)
plt.plot(data21['r']/h, data21['rho']*h*h, '--', linewidth=2, color=colorlist[1], label=r'0.05 Gyr', alpha=0.7)
#plt.plot(data22['r']/h, data22['rho']*h*h, '--', linewidth=2, color=colorlist[3], label=r'0.4 yr', alpha=0.7)
plt.plot(data23['r']/h, data23['rho']*h*h, '--', linewidth=2, color=colorlist[5], label=r'0.1 Gyr', alpha=0.7)
plt.plot(data24['r']/h, data24['rho']*h*h, '--', linewidth=2, color=colorlist[7], label=r'0.15 Gyr', alpha=0.7)
#plt.plot(data25['r']/h, data25['rho']*h*h, '--', linewidth=2, color=colorlist[8], label=r'$6.6 Gyr (0.2t_{\rm cb})$', alpha=0.7)
#plt.plot(data26['r']/h, data26['rho']*h*h, '--', linewidth=2, color=colorlist[8], label=r'10 Gyr', alpha=0.7)
#plt.plot(data27['r']/h, data27['rho']*h*h, '--', linewidth=2, color=colorlist[9], label=r'12 Gyr', alpha=0.7)



plt.ylim([1e5,1e10])
plt.xlim([0.1, 100])

plt.xscale('log')
plt.yscale('log')

plt.legend(ncol=2)
plt.show()
fig.savefig('NH2_sidm_profile.pdf', bbox_inches='tight')

