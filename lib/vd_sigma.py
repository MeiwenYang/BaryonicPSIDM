from scipy.misc import derivative
import numpy as np
import os
import glob
from functools import partial
import numpy as np
from mpmath import polylog
from numba import njit
from scipy import integrate
from astropy import units as ut
from astropy import constants as ct
from scipy.special import exp1, gammaincc
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
import pandas as pd


from matplotlib import pyplot as plt
#%matplotlib notebook
#figsize = (12,9)
figsize = None
#%matplotlib inline
import vegas
import multiprocessing
from scipy import interpolate

from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#import seaborn as sns
from scipy import constants
import numpy as np


G= 6.673e-11 * (1.9891e30*3.1536e7**2/3.085678e16**3) # pc3/Msun/yr2
Mspc3_GeVcm3 = 1./0.02633906747374293 # GeV/cm3 / (Ms/pc^3) #

def frho_nfw(rhos,rs,r):
    return rhos/(r/rs)/pow(1+r/rs,2)

def fHern(rhoH,rH,r):
   if rH>0: 
       #print("r,rH",r,rH) 
       return rhoH/((r/rH)*pow(1+r/rH,3))
   else:
       return 0

def frho_total(rhos,rs,rhoH,rH,r):
    return frho_nfw(rhos,rs,r)+fHern(rhoH,rH,r)

def MtotNFW(r,rhoss,rss):
   #if(r<=0): return 0
   return 4*np.pi*rhoss*pow(rss,3)*(-1.0 + rss/(r + rss) - np.log(rss) + np.log(r + rss))

def MrHern(r,rhoH,rH):
   return 2*np.pi*pow(r,2)*rhoH*pow(rH,3)/pow(r + rH,2)

def M_tot(rhos,rs,rhoH,rH,r):
   return MtotNFW(r,rhos,rs)+MrHern(r,rhoH,rH)

def pressure(rhos,rs,rhoH,rH,r):
    rmax=1e9
    p_integrand = lambda lnr: G*M_tot(rhos,rs,rhoH,rH,np.exp(lnr))*frho_nfw(rhos,rs,np.exp(lnr))/(np.exp(lnr)) #pc2/yr2
    return integrate.quad(p_integrand,np.log(r),np.log(rmax))[0]

def sigma2(rhos,rs,rhoH,rH,r):
    return pressure(rhos,rs,rhoH,rH,r)/frho_nfw(rhos,rs,r)*9.77e5**2/1e3
        


def dsigma_dr_numerical(rhoH_rhos, rH_rs):
    # 定义只依赖半径的包装函数
    rhos=1
    rs=1
    #r=0.5
    r=0.05*rs
    #r=0.49*rs*Ft(rhoH_rhos,rH_rs)**2
    #print("r/rs",r/rs)
    #r=0.05*rs
    #r=1
    rhoH=rhoH_rhos*rhos
    rH=rH_rs*rs
    def func(rr):
        return np.sqrt(sigma2(rhos, rs, rhoH, rH, rr))
    
    # 三阶差分，相对步长 1e-4
    return derivative(func, r, dx=r*1e-4, order=3)/np.sqrt(sigma2(rhos, rs, rhoH, rH, r))*r
