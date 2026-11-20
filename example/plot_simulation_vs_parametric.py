import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.vd_sigma import *
from lib.parametricCoredDZ_legacy import *
from lib.parametricCoredDZ import *

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


PROFILE_DIR = Path(__file__).resolve().parent / "controlled_simulation_profiles"



def load_and_interpolate(rho_file):
    data = np.genfromtxt(rho_file, usecols=range(10), names=True)
    frho = interp1d(data['r'], data['rho'], fill_value='extrapolate')
    return frho


def read_density_profile(filename):
    data = np.loadtxt(
        filename,
        comments="#",
        usecols=(0, 1)
    )

    radius_center = data[:, 0]  # kpc
    density_shell = data[:, 1]  # Msun/kpc^3

    mask = (
        np.isfinite(radius_center)
        & np.isfinite(density_shell)
        & (radius_center > 0)
    )

    radius_center = radius_center[mask]
    density_shell = density_shell[mask]

    order = np.argsort(radius_center)

    return radius_center[order], density_shell[order]


def recover_edges_from_centers(radius_center):
    radius_center = np.asarray(radius_center, dtype=float)

    edges = np.zeros(len(radius_center) + 1)

    for i, center in enumerate(radius_center):
        edges[i + 1] = 2.0 * center - edges[i]


    return edges


def mean_density_from_density_profile(
    filename,
    target_radius
):

    radius_center, density_shell = read_density_profile(filename)
    #print("filename",filename)
    edges = recover_edges_from_centers(radius_center)

    R = float(target_radius)

    if R <= 0:
        raise ValueError("target_radius must be >0")

    if R > edges[-1]:
        raise ValueError(
            f"target_radius {R} kpc is out of profile range "
            f"{edges[-1]:.6g} kpc"
        )

    enclosed_mass = 0.0

    for i, rho_i in enumerate(density_shell):
        r_inner = edges[i]
        r_outer = edges[i + 1]

        if R <= r_inner:
            break

        integration_outer = min(R, r_outer)

        shell_volume = (
            4.0 * np.pi / 3.0
            * (
                integration_outer**3
                - r_inner**3
            )
        )

        enclosed_mass += rho_i * shell_volume

        if R <= r_outer:
            break

    sphere_volume = (
        4.0 * np.pi / 3.0
        * R**3
    )

    return enclosed_mass / sphere_volume


# ========== Old simulation benchmarks arxiv:2405.03787 ==========

TIME_RULES = {
    "old_dm11_only": (0.25, 0.0),       # t = 0.25 * snapshot
    "old_dm11_baryon": (0.25, 0.0),     # t = 0.25 * snapshot
    "old_dm11_baryon_r2": (0.25, -0.25),# t = 0.25 * (snapshot - 1)
    "old_dm11_baryon_m2": (0.25, -0.25),# t = 0.25 * (snapshot - 1)
    "old_dm12_mw": (0.10, 0.0),         # t = 0.10 * snapshot
}




TIME_RULES = {
    "old_dm11_only": (0.25, 0.0),        # t = 0.25 * snapshot
    "old_dm11_baryon": (0.25, 0.0),      # t = 0.25 * snapshot
    "old_dm11_baryon_r2": (0.25, -0.25), # t = 0.25 * (snapshot - 1)
    "old_dm11_baryon_m2": (0.25, -0.25), # t = 0.25 * (snapshot - 1)
    "old_dm12_mw": (0.10, 0.0),          # t = 0.10 * snapshot
}


FILE_TEMPLATES = {
    "old_dm11_only": str(PROFILE_DIR / "old_dm11_only/prof_bound_halo_prof_run1_bm1_xs10_snap{snapshot:03d}.txt"),
    "old_dm11_baryon": str(PROFILE_DIR / "old_dm11_baryon/prof_bound_halo_prof_run1_bm2_xs10_baryon_snap{snapshot:03d}.txt"),
    "old_dm11_baryon_r2": str(PROFILE_DIR / "old_dm11_baryon_r2/prof_bound_halo_prof_run1_bm4_xs10_baryon_snap{snapshot:03d}.txt"),
    "old_dm11_baryon_m2": str(PROFILE_DIR / "old_dm11_baryon_m2/prof_bound_halo_prof_run1_bm7_xs10_doubleB_snap{snapshot:03d}.txt"),
    "old_dm12_mw": str(PROFILE_DIR / "old_dm12_mw/prof_bound_halo_prof_sidm1_IC5Gyr_eps015_MW_run1_MWscale_median_snap{snapshot:03d}.txt"),
}


SNAPSHOT_COUNTS = {
    "old_dm11_only": 129,       
    "old_dm11_baryon": 62,     
    "old_dm11_baryon_r2": 150,  
    "old_dm11_baryon_m2": 55,
    "old_dm12_mw": 80,      
}

################## scale radii ########################
SCALE = 0.1
RS_VALUES = {
    "old_dm11_only": 9.1,
    "old_dm11_baryon": 9.1,
    "old_dm11_baryon_r2": 9.1,
    "old_dm11_baryon_m2": 9.1,
    "old_dm12_mw": 15.5453, 
}


# ========== New simulation benchmarks ==========

TIME_RULES.update({
    "BM10_c": (0.20, 0.0),
    "BM11_c": (0.05, 0.0),
    "BM12_c": (0.15, 0.0),
    "BM11": (0.25, 0.0),
    "BM12": (0.05, 0.0),
    "BM13":(0.25,0.0),
})


FILE_TEMPLATES.update({
    "BM10_c": str(PROFILE_DIR / "BM10_c/prof_bound_halo_snapshot_snap{snapshot:03d}.txt"),
    "BM11_c": str(PROFILE_DIR / "BM11_c/prof_bound_halo_snapshot_snap{snapshot:03d}.txt"),
    "BM12_c": str(PROFILE_DIR / "BM12_c/prof_bound_halo_snapshot_snap{snapshot:03d}.txt"),
    "BM11": str(PROFILE_DIR / "BM11/prof_bound_halo_snapshot_snap{snapshot:03d}.txt"),
    "BM12": str(PROFILE_DIR / "BM12/prof_bound_halo_snapshot_snap{snapshot:03d}.txt"),
    "BM13": str(PROFILE_DIR / "BM13/prof_bound_halo_snapshot_snap{snapshot:03d}.txt"),
})

SNAPSHOT_COUNTS.update({
    "BM10_c": 10,   
    "BM11_c": 200,   
    "BM12_c": 20,   
    "BM11": 30,    
    "BM12": 130,
    "BM13": 52,    
})

RS_VALUES.update({
    "BM10_c": 3.8,
    "BM11_c": 7,    
    "BM12_c": 17.5,     
    "BM11": 15.6,        
    "BM12": 24, 
    "BM13": 50.4,
})

import numpy as np

# ============================================================

# rho_s : Msun / kpc^3
# r_s   : kpc
# rhoH  : Msun / kpc^3
# rH    : kpc
# tc/tcb: Gyr
#
# tau
# DM-only: tau = t / tc
# DM+Baryon:  tau = t / tcb
# ============================================================


# ============================================================
# 1. 旧 DM11 only / BM1 / SIDM10
# ============================================================

rhos_BM11_DMO = 6.890540000000000e6
rs_BM11_DMO = 9.10468
R200_BM11_DMO = 127.634
c200_BM11_DMO = R200_BM11_DMO / rs_BM11_DMO

tc_BM11_DMO = 65.86912110925157  # Gyr


# ============================================================
# 2. 旧 DM11+baryon / BM2 / SIDM10
# ============================================================

rhos_BM11_B = 6.890540000000000e6
rs_BM11_B = 9.10468
R200_BM11_B = 127.634
c200_BM11_B = R200_BM11_B / rs_BM11_B
rhoH_BM2 = 3.563527995543644e8
rH_BM2 = 0.767986389328751

tcb_BM11_B = 25.343761928125183  # Gyr

# 保留你原来的变量名，但它物理上代表 t_cb
tc_BM11_B = tcb_BM11_B


# ============================================================
# 3. 旧 DM11+baryon R2 / BM4 / SIDM10
# ============================================================

rhos_BM11_R2 = 6.890540000000000e6
rs_BM11_R2 = 9.10468
R200_BM11_R2 = 127.634
c200_BM11_R2 = R200_BM11_R2 / rs_BM11_R2

rhoH_BM4 = 8.951177620310610e7
rH_BM4 = 1.2171764003499277

tcb_BM11_R2 = 32.20788474621624  # Gyr
tc_BM11_R2 = tcb_BM11_R2


# ============================================================
# 4. 旧 DM11+baryon M2 / BM7 / SIDM10
# ============================================================

rhos_BM11_M2 = 6.890540000000000e6
rs_BM11_M2 = 9.10468
R200_BM11_M2 = 127.634
c200_BM11_M2 = R200_BM11_M2 / rs_BM11_M2


rhoH_BM7 = 5.342100331605571e8
rH_BM7 = 0.8454467001724245

tcb_BM11_M2 = 11.89224054624407  # Gyr
tc_BM11_M2 = tcb_BM11_M2


# ============================================================
# 5. 旧 DM12+baryon / MW / SIDM1
# ============================================================

rhos_BM12_MW = 1.2260408e7
rs_BM12_MW = 15.5453
R200_BM12_MW = 206.178
c200_BM12_MW = R200_BM12_MW / rs_BM12_MW

rhoH_MW = 1.83155e9
rH_MW = 1.43878

tcb_BM12_MW = 8.954686877236082  # Gyr
tc_BM12_MW = tcb_BM12_MW



# ============================================================
# BM13
# ============================================================
c200_BM13 = 6.9
rhos_BM13 = 4.3e6
rs_BM13 = 50.4
rhoH_BM13 = rhos_BM13*235.2
rH_BM13 = 0.07*rs_BM13


BM13_tcb = tcb(1.65, rhos_BM13, rs_BM13, rhoH_BM13, rH_BM13) 
BM13_tc  = BM13_tcb

# ============================================================
# BM12_c
# ============================================================
c200_BM12_c = 15.67
rhos_BM12_c = 8914384.2927
rs_BM12_c = 17.56
rhoH_BM12_c = 1090202292.0413
rH_BM12_c = 1.75


BM12_c_tcb = tcb(4.5, rhos_BM12_c, rs_BM12_c, rhoH_BM12_c, rH_BM12_c) 
BM12_c_tc  = BM12_c_tcb


# ============================================================
# BM10_c
# ============================================================
c200_BM10_c = 27.06
rhos_BM10_c = 36977584.43603558
rs_BM10_c = 3.8
rhoH_BM10_c = 29664251.431 * 2**3          # 约 2.37e8
rH_BM10_c = 2.67 / 1.9                     # 约 1.4053


BM10_c_tcb = tcb(8, rhos_BM10_c, rs_BM10_c, rhoH_BM10_c, rH_BM10_c) 
BM10_c_tc  = BM10_c_tcb


# ============================================================
# BM11c
# ============================================================
c200_BM11_c = 17.47
rhos_BM11_c = 1.17e7
rs_BM11_c = 7.0
rhoH_BM11_c = 1.17e7 * 52.7               # 约 6.166e8
rH_BM11_c = 7.0 * 0.1                     # 0.7


BM11_c_tcb = tcb(9, rhos_BM11_c, rs_BM11_c, rhoH_BM11_c, rH_BM11_c) 
BM11_c_tc  = BM11_c_tcb


############### BM12 ###################
c200_BM12 = 10.9
rhos_BM12 = 3.6e6
rs_BM12 = 24.0
rhoH_BM12 = rhos_BM12 * 213          # 约 7.668e8
rH_BM12 = rs_BM12 * 0.0799           # 约 1.9176

BM12_tcb = tcb(5, rhos_BM12, rs_BM12, rhoH_BM12, rH_BM12)
BM12_tc  = BM12_tcb


############ BM11 #######################
c200_BM11 = 219.8 / 15.58
rhos_BM11 = 6790343
rs_BM11 = 15.58
rhoH_BM11 = 0.347e10
rH_BM11 = 0.651

BM11_tcb = tcb(6.5, rhos_BM11, rs_BM11, rhoH_BM11, rH_BM11)
BM11_tc  = BM11_tcb


#############################################
# 存储结果的字典
results = {}

for case, (dt, offset) in TIME_RULES.items():
    times = []
    rhos = []
    file_template = FILE_TEMPLATES[case]
    n_snap = SNAPSHOT_COUNTS[case]
    rs = RS_VALUES[case]
    
    target_radius = SCALE * rs  

    for snap in range(n_snap):
        if case in ["old_dm11_baryon_r2", "old_dm11_baryon_m2"]:
            snap_idx = snap + 1
        else:
            snap_idx = snap

        filename = file_template.format(snapshot=snap_idx)
        try:
            frho = load_and_interpolate(filename)
        except Exception as e:
            #print(f"Warning: Failed to load {filename}: {e}")
            continue

        t = dt * snap + offset
        rho = mean_density_from_density_profile(filename,target_radius) #frho(target_radius)

        times.append(t)
        rhos.append(rho)

    results[case] = {
        "time": np.array(times),
        "rho": np.array(rhos)
    }
    print(f"Case {case}: loaded {len(times)} snapshots, rho range: {np.min(rhos):.2e} - {np.max(rhos):.2e}")


array_t_BM11_DMO = results["old_dm11_only"]["time"]
array_rho_BM11_DMO = results["old_dm11_only"]["rho"]

array_t_BM11_B = results["old_dm11_baryon"]["time"]
array_rho_BM11_B = results["old_dm11_baryon"]["rho"]

array_t_BM11_r2=results["old_dm11_baryon_r2"]["time"]
array_rho_BM11_r2 =results["old_dm11_baryon_r2"]["rho"]


array_t_BM11_m2=results["old_dm11_baryon_m2"]["time"]
array_rho_BM11_m2 =results["old_dm11_baryon_m2"]["rho"]

array_t_BM12_mw = results["old_dm12_mw"]["time"]
array_rho_BM12_mw = results["old_dm12_mw"]["rho"]


array_t_BM10_c = results["BM10_c"]["time"]
array_rho_BM10_c = results["BM10_c"]["rho"]

array_t_BM11_c = results["BM11_c"]["time"]
array_rho_BM11_c = results["BM11_c"]["rho"]

array_t_BM12_c = results["BM12_c"]["time"]
array_rho_BM12_c = results["BM12_c"]["rho"]

array_t_BM11 = results["BM11"]["time"]
array_rho_BM11 = results["BM11"]["rho"]

array_t_BM12 = results["BM12"]["time"]
array_rho_BM12 = results["BM12"]["rho"]

array_t_BM13 = results["BM13"]["time"]
array_rho_BM13 = results["BM13"]["rho"]



radius_DM11 = SCALE*9.1  # kpc，用于 DM11 系列
radius_DM12 = SCALE*15.5  # kpc，用于 DM12 MW

Mh_BM11=1e11
Mh_BM12=1e12


BM11_DMO_PM = np.array([
    rhocbDZ(radius_DM11, Mh_BM11, 0.0, rhos_BM11_DMO, c200_BM11_DMO, rs_BM11_DMO, t/tc_BM11_DMO, 0.0, 1.0)
    for t in array_t_BM11_DMO
])

BM11_B_PM = np.array([
    rhocbDZ(radius_DM11, Mh_BM11, 0.01*Mh_BM11, rhos_BM11_B, c200_BM11_B, rs_BM11_B, t/tc_BM11_B, rhoH_BM2, rH_BM2)
    for t in array_t_BM11_B
])

BM11_R2_PM = np.array([
    rhocbDZ(radius_DM11, Mh_BM11, 0.01*Mh_BM11, rhos_BM11_R2, c200_BM11_R2, rs_BM11_R2, t/tc_BM11_R2, rhoH_BM4, rH_BM4)
    for t in array_t_BM11_r2
])

BM11_M2_PM = np.array([
    rhocbDZ(radius_DM11, Mh_BM11, 0.02*Mh_BM11, rhos_BM11_M2, c200_BM11_M2, rs_BM11_M2, t/tc_BM11_M2, rhoH_BM7, rH_BM7)
    for t in array_t_BM11_m2
])

BM12_MW_PM = np.array([
    rhocbDZ(radius_DM12, Mh_BM12, 0.034*Mh_BM12, rhos_BM12_MW, c200_BM12_MW, rs_BM12_MW, t/tc_BM12_MW, rhoH_MW, rH_MW)
    for t in array_t_BM12_mw
])


# ============================================================
# BM12_c
# ============================================================

heps_BM12_c = SCALE * rs_BM12_c 
Mh_BM12_c   = 1.3e12
Ms_BM12_c   = 3.7e10

BM12_c_PM = np.array([
    rhocbDZ(heps_BM12_c, Mh_BM12_c, Ms_BM12_c, rhos_BM12_c, c200_BM12_c, rs_BM12_c, t/BM12_c_tc, rhoH_BM12_c, rH_BM12_c)
    for t in array_t_BM12_c
])




# ============================================================
# BM10_c
# ============================================================
heps_BM10_c = SCALE * rs_BM10_c         
Mh_BM10_c   = 6.1e10                    
Ms_BM10_c   = 4.1e9                     

BM10_c_PM = np.array([
    rhocbDZ(heps_BM10_c, Mh_BM10_c, Ms_BM10_c, rhos_BM10_c, c200_BM10_c, rs_BM10_c, t/BM10_c_tc, rhoH_BM10_c, rH_BM10_c)
    for t in array_t_BM10_c
])

# ============================================================
# BM11c
# ============================================================
heps_BM11_c = SCALE * rs_BM11_c       
Mh_BM11_c   = 1.e11                     
Ms_BM11_c   = 1.4e9            

BM11_c_PM = np.array([
    rhocbDZ(heps_BM11_c, Mh_BM11_c, Ms_BM11_c, rhos_BM11_c, c200_BM11_c, rs_BM11_c, t/BM11_c_tc, rhoH_BM11_c, rH_BM11_c)
    for t in array_t_BM11_c
])


# ============================================================
# BM11
# ============================================================
heps_BM11 = SCALE * rs_BM11             
Mh_BM11   = 5.7e11                      
Ms_BM11   = 6e9                      

BM11_PM = np.array([
    rhocbDZ(heps_BM11, Mh_BM11, Ms_BM11, rhos_BM11, c200_BM11, rs_BM11, t/BM11_tc, rhoH_BM11, rH_BM11)
    for t in array_t_BM11
])


# ============================================================
# BM12
# ============================================================
heps_BM12 = SCALE * rs_BM12         
Mh_BM12   = 1.e12                      
Ms_BM12   = 3.4e10                      

BM12_PM = np.array([
    rhocbDZ(heps_BM12, Mh_BM12, Ms_BM12, rhos_BM12, c200_BM12, rs_BM12, t/BM12_tc, rhoH_BM12, rH_BM12)
    for t in array_t_BM12
])

#print("BM12_tau",array_t_BM12/BM12_tc)
# ============================================================
# BM13
# ============================================================
heps_BM13 = SCALE * rs_BM13       
Mh_BM13   = 1.3e13                      
Ms_BM13   = 2.5e11                      


BM13_PM = np.array([
    rhocbDZ(heps_BM13, Mh_BM13, Ms_BM13, rhos_BM13, c200_BM13, rs_BM13, t/BM13_tc, rhoH_BM13, rH_BM13)
    for t in array_t_BM13
])

#print("BM13_tau",array_t_BM13/BM13_tc)


#############################################################

print("benchmark: Mh, Ms, rhos, c200, rs, rhoH, rH")

print("BM10_c:", Mh_BM10_c, Ms_BM10_c, rhos_BM10_c,
      c200_BM10_c, rs_BM10_c, rhoH_BM10_c, rH_BM10_c)

print("BM11_c:", Mh_BM11_c, Ms_BM11_c, rhos_BM11_c,
      c200_BM11_c, rs_BM11_c, rhoH_BM11_c, rH_BM11_c)

print("BM12_c:", Mh_BM12_c, Ms_BM12_c, rhos_BM12_c,
      c200_BM12_c, rs_BM12_c, rhoH_BM12_c, rH_BM12_c)

print("BM11:", Mh_BM11, Ms_BM11, rhos_BM11,
      c200_BM11, rs_BM11, rhoH_BM11, rH_BM11)

print("BM12:", Mh_BM12, Ms_BM12, rhos_BM12,
      c200_BM12, rs_BM12, rhoH_BM12, rH_BM12)

print("BM13:", Mh_BM13, Ms_BM13, rhos_BM13,
      c200_BM13, rs_BM13, rhoH_BM13, rH_BM13)


###########################################################

import matplotlib.pyplot as plt
import numpy as np

colors = {
    # ---- old benchmarks ----
    'DM11_only': '#4393E6',       # sky blue
    'DM11_baryon': '#F06A91',     # rose pink
    'DM11_r2': '#22B8C5',         # turquoise
    'DM11_m2': '#F39C6B', 
    'DM12_mw': '#8D7AE6',         # lavender

    # ---- new benchmarks ----
    'DM13_compact': (255/255, 36/255, 33/255),   # #FF2421
    'DM12': '#F28522',                           # orange-brown
    'DM11c': (59/255, 163/255, 116/255),          # #3BA374
    'DM12_compact': (70/255, 69/255, 232/255),   # #4645E8
    'DM11': '#AF58BA',                           # purple
    'DM10': 'grey'                               # #808080
}


labels = {
    # ---- old benchmarks ----
    'DM11_only': 'DM11 only',
    'DM11_baryon': 'DM11+baryon',
    'DM11_r2': 'DM11+baryon R2',
    'DM11_m2': 'DM11+baryon M2',
    'DM12_mw': 'DM12+baryon (MW)',

    # ---- new benchmarks ----
    'DM13_compact': 'DM13 compact',
    'DM11c': 'DM11c',
    'DM12': 'DM12',
    'DM12_compact': 'DM12 compact',
    'DM11': 'DM11',
    'DM10': 'DM10'
}

data = [
    (array_t_BM11_DMO, array_rho_BM11_DMO, 'DM11_only'),
    (array_t_BM11_B, array_rho_BM11_B, 'DM11_baryon'),
    (array_t_BM11_r2, array_rho_BM11_r2, 'DM11_r2'),
    (array_t_BM11_m2, array_rho_BM11_m2, 'DM11_m2'),
    (array_t_BM12_mw, array_rho_BM12_mw, 'DM12_mw'),

    (array_t_BM10_c, array_rho_BM10_c, 'DM10'),
    (array_t_BM11_c, array_rho_BM11_c, 'DM11c'),
    (array_t_BM12_c, array_rho_BM12_c, 'DM12_compact'),
    (array_t_BM11, array_rho_BM11, 'DM11'),          
    (array_t_BM12, array_rho_BM12, 'DM12'),          
    (array_t_BM13, array_rho_BM13, 'DM13_compact'),  
]

plt.figure(figsize=(6, 6))


for t, rho, key in data:
    plt.plot(t, rho,
             color=colors[key],
             linewidth=10,
             alpha=0.2,
             linestyle='-',
             label=labels[key])

# PM 
plt.plot(array_t_BM11_DMO, BM11_DMO_PM, color=colors['DM11_only'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM11_B, BM11_B_PM, color=colors['DM11_baryon'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM11_r2, BM11_R2_PM, color=colors['DM11_r2'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM11_m2, BM11_M2_PM, color=colors['DM11_m2'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM12_mw, BM12_MW_PM, color=colors['DM12_mw'], linewidth=3, alpha=1.0)

plt.plot(array_t_BM10_c, BM10_c_PM,color=colors['DM10'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM11_c, BM11_c_PM, color=colors['DM11c'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM12_c, BM12_c_PM, color=colors['DM12_compact'], linewidth=3, alpha=1.0)
plt.plot(array_t_BM11, BM11_PM, color=colors['DM11'],linewidth=3, alpha=1.0)
plt.plot(array_t_BM12, BM12_PM, color=colors['DM12'],linewidth=3, alpha=1.0)
plt.plot(array_t_BM13, BM13_PM, color=colors['DM13_compact'],linewidth=3, alpha=1.0)

plt.text(
    0.5, 0.85,
    "Thick lines: simulations\n"
    "Thin lines: Recalibrated PSIDM",
    transform=plt.gca().transAxes,
    fontsize=11,
    ha='left', va='bottom',
    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8)
)

plt.xlim(0, 50)
plt.ylim(1e5,1e10)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$t \ \mathrm{(Gyr)}$', fontsize=18)
plt.ylabel(r'$\bar{\rho} \ \mathrm{(M_\odot / kpc^3)}$', fontsize=18)
plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.legend(fontsize=11, loc='best',ncol=2)
plt.title(r'$r_{\rm inner}=0.1\,r_s$', fontsize=16)

plt.tight_layout()
plt.savefig('simulation_vs_parametric.png', bbox_inches='tight')
plt.show()


