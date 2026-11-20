import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.patches as mpatches
import math
from math import log
from math import sqrt
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.modeling.physical_models import NFW
import astropy.constants as c
import astropy.units as u
#import pynbody as nb
import matplotlib.cm as cm
from parametricCoredDZ import *

kpc2_cm2=(3.085678e19*1e2)**2
Msun_g=1.9891e30*1e3


def sigma_m(rhos,rs):
    return 0.2/(rhos*rs)*kpc2_cm2/Msun_g  #kpc2/Msun


c200=219.8/15.58
rhos=6790343
rs=15.58
rhoH=0.347e10
rH=0.651
BM_tcb=tcb(sigma_m(rhos,rs),rhos,rs,rhoH,rH)*1.1
print("tcb",BM_tcb)
fr=np.logspace(np.log10(0.1),np.log10(500),20)
fstar=np.array([fHern(r, rhoH,rH) for r in fr])
BM_0=np.array([rhoDZcore(r,rhos,c200,rs,0.0,rhoH,rH) for r in fr])
BM_0d2=np.array([rhoDZcore(r,rhos,c200,rs,0.2/BM_tcb,rhoH,rH) for r in fr])
BM_0d4=np.array([rhoDZcore(r,rhos,c200,rs,2.8/BM_tcb,rhoH,rH) for r in fr])
BM_0d6=np.array([rhoDZcore(r,rhos,c200,rs,4.2/BM_tcb,rhoH,rH) for r in fr])
BM_1=np.array([rhoDZcore(r,rhos,c200,rs,6.8/BM_tcb,rhoH,rH) for r in fr])
BM_0d8=np.array([rhoDZcore(r,rhos,c200,rs,5.6/BM_tcb,rhoH,rH) for r in fr])


data10 = np.genfromtxt('prof_bound_halo_snapshot_snap000.txt', names=True)
data11 = np.genfromtxt('prof_bound_halo_snapshot_snap001.txt', names=True)
#data12 = np.genfromtxt('prof_bound_halo_snapshot_snap004.txt', names=True)
data13 = np.genfromtxt('prof_bound_halo_snapshot_snap014.txt', names=True)
data14 = np.genfromtxt('prof_bound_halo_snapshot_snap021.txt', names=True)
data15 = np.genfromtxt('prof_bound_halo_snapshot_snap034.txt', names=True)
data16 = np.genfromtxt('prof_bound_halo_snapshot_snap028.txt', names=True)
data17 = np.genfromtxt('prof_bound_halo_snapshot_snap000.txt', names=True)

data20 = np.genfromtxt('prof_bound_stars_snapshot_snap000.txt', names=True)
data21 = np.genfromtxt('prof_bound_stars_snapshot_snap001.txt', names=True)
#data12 = np.genfromtxt('prof_bound_halo_snapshot_snap004.txt', names=True)
data23 = np.genfromtxt('prof_bound_stars_snapshot_snap014.txt', names=True)
data24 = np.genfromtxt('prof_bound_stars_snapshot_snap021.txt', names=True)
data25 = np.genfromtxt('prof_bound_stars_snapshot_snap034.txt', names=True)
data26 = np.genfromtxt('prof_bound_stars_snapshot_snap028.txt', names=True)
data27 = np.genfromtxt('prof_bound_stars_snapshot_snap000.txt', names=True)


h = 1

fig = plt.figure()
plt.xlabel(r"$r\ ({\rm kpc})$",fontsize=12)
plt.ylabel(r"$\rho \ (M_{\odot} {\rm kpc}^{-3}) $",fontsize=12)

cmap = cm.get_cmap('coolwarm')
colorlist = cmap(np.linspace(0, 1, 10))

def rhoNFW(r,rhos,rs):
   return rhos/(r/rs)/pow(1+r/rs,2)

# 创建 2x2 子图网格，正确解包二维结构
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
ax1, ax2,ax3 = axes[0]  # 第一行的两个子图
ax4, ax5,ax6 = axes[1]  # 第二行的两个子图

# --- 子图1: data10 vs BM_0 ---
ax1.plot(data10['r']/h, (data10['rho'])*h*h, linewidth=5, color=colorlist[0], label=r'$0 \rm Gyr \; \rm Nbody$', alpha=0.5)
ax1.plot(data20['r']/h, (data20['rho'])*h*h, linewidth=5, color='black', label=r'$\rm star\; \rm Nbody$', alpha=0.5)
ax1.plot(fr, BM_0, linestyle='-.', color=colorlist[0], linewidth=2, label='parametric model')

# --- 子图2: data11 vs BM_0d2 ---
ax2.plot(data11['r']/h, (data11['rho'])*h*h, linewidth=5, color=colorlist[3], label=r'$0.2t_{\rm cb} \; \rm Nbody$', alpha=0.5)
ax2.plot(data21['r']/h, (data21['rho'])*h*h, linewidth=5, color='black', label=r'$\rm star\; \rm Nbody$', alpha=0.5)
ax2.plot(fr, BM_0d2, linestyle='-.', color=colorlist[3], linewidth=2,label='parametric model')

# --- 子图3: data13 vs BM_0d4 ---
ax3.plot(data13['r']/h, (data13['rho'])*h*h, linewidth=5, color=colorlist[5], label=r'$0.4t_{\rm cb} \; \rm Nbody$', alpha=0.5)
ax3.plot(data23['r']/h, (data23['rho'])*h*h, linewidth=5, color='black', label=r'$\rm star\; \rm Nbody$', alpha=0.5)
ax3.plot(fr, BM_0d4, linestyle='-.', color=colorlist[5], linewidth=2, label='parametric model')

# --- 子图4: data14 vs BM_0dd ---
ax4.plot(data14['r']/h, (data14['rho'])*h*h, linewidth=5, color=colorlist[7], label=r'$0.6t_{\rm cb} \; \rm Nbody$', alpha=0.5)
ax4.plot(data24['r']/h, (data24['rho'])*h*h, linewidth=5, color='black', label=r'$\rm star\; \rm Nbody$', alpha=0.5)
ax4.plot(fr, BM_0d6, linestyle='-.', color=colorlist[7], linewidth=2, label='parametric model')

# --- 子图4: data14 vs BM_0dd ---
ax5.plot(data16['r']/h, (data16['rho'])*h*h, linewidth=5, color=colorlist[8], label=r'$0.8t_{\rm cb} \; \rm Nbody$', alpha=0.5)
ax5.plot(data26['r']/h, (data26['rho'])*h*h, linewidth=5, color='black', label=r'$\rm star\; \rm Nbody$', alpha=0.5)
ax5.plot(fr, BM_0d8, linestyle='-.', color=colorlist[8], linewidth=2, label='parametric model')

# --- 子图6: data15 vs BM_1 ---
ax6.plot(data15['r']/h, (data15['rho'])*h*h, linewidth=5, color=colorlist[9], label=r'$t_{\rm cb} \; \rm Nbody$', alpha=0.7)
ax6.plot(data25['r']/h, (data25['rho'])*h*h, linewidth=5, color='black', label=r'$\rm star\; \rm Nbody$', alpha=0.5)
ax6.plot(fr, BM_1, linestyle='-.', color=colorlist[9], linewidth=2, label='parametric model')

# 在所有子图中添加 fstar 曲线（可选）
for ax in [ax1, ax2, ax3, ax4,ax5,ax6]:
    ax.plot(fr, fstar, linestyle='-.', color='black', linewidth=1, label='star')

# ---------------------------
# 添加坐标轴标签和标题
# ---------------------------
# 1. 设置统一的 X/Y 轴标签
for ax in [ax1, ax2, ax3, ax4,ax5,ax6]:
    ax.set_xlabel(r'$r\ ({\rm kpc})$', fontsize=12)          # X轴标签
    ax.set_ylabel(r'$\rho \ (M_{\odot} {\rm kpc}^{-3})$', fontsize=12)     # Y轴标签（单位根据实际情况调整）

# 2. 添加总标题（固定重子势）
fig.suptitle(r'$m_{\rm star}=4 \;m_{\rm dm}\;\;\rm Baryon \;(live\; particle) $', fontsize=16, y=1.02)  # y参数调整标题垂直位置

# 统一设置坐标轴属性
for ax in [ax1, ax2, ax3, ax4,ax5,ax6]:
    ax.set_ylim([1e5, 5e9])
    ax.set_xlim([0.25, 50])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(ncol=2, fontsize=12)  # 缩小图例字体

plt.tight_layout()
fig.savefig('sidm_profile.pdf', bbox_inches='tight')
plt.show()
