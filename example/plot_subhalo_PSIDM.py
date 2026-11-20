import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from lib.parametricCoredDZ import *
from lib.parametricCoredDZ import _legacy_recipe_a0_c0
import lib.parametricCoredDZ_legacy as _legacy


highlight_ids = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    15, 16, 17, 19, 23, 24, 26, 29, 34, 38, 46, 58
}

# ============================================================
# 文件与画图设置
# ============================================================
CSV_FILE = "PM_data_main0_w_120_sigma0_70.csv"

# CSV 中 Mh、Mb 的单位为 1e10 Msun
MASS_UNIT = 1.0e10

# 公共归一化半径网格
x_grid = np.logspace(-3.5, -0.2, 100)

# 输出文件
OUTPUT_PDF = "subhalo_profiles.png"

# ============================================================
# Recalibrated PSIDM
# ============================================================

def rho_hybrid(row, r):
    Mh = float(row["Mh"]) * MASS_UNIT
    Ms = float(row["Mb"]) * MASS_UNIT

    tr = float(row["tr"])

    rhosx = float(row["rhosx"])
    rsx = float(row["rsx"])

    rhoss = float(row["rhoss"])
    rss = float(row["rss"])

    rhoH = float(row["rhoH"])
    rH = float(row["rH"])

    c200 = float(row["c200"])
    cvir = float(row["cvir"])

    r = np.asarray(r, dtype=float)
    rho= np.array([rhoDZcore_cal(Mh,Ms,rr,rhosx,c200,rsx,tr,rhoH,rH,rhoss,rss) for rr in r], dtype=float)

    return rho


# ============================================================
# original PSIDM
# ============================================================
def rho_original_parametric(row, r):
    Mh = float(row["Mh"]) * MASS_UNIT
    Ms = float(row["Mb"]) * MASS_UNIT

    tr = float(row["tr"])

    rhosx = float(row["rhosx"])
    rsx = float(row["rsx"])

    rhoss = float(row["rhoss"])
    rss = float(row["rss"])

    rhoH = float(row["rhoH"])
    rH = float(row["rH"])

    c200 = float(row["c200"])
    cvir = float(row["cvir"])

    r = np.asarray(r, dtype=float)
    rho = np.array([_legacy.rhoDZcore(Mh,Ms,rr,rhosx,c200,rsx,tr,rhoH,rH) for rr in r], dtype=float)

    return rho


# ============================================================
# CDM DZ profile
# ============================================================

def rho_CDM(row, r):
    Mh = float(row["Mh"]) * MASS_UNIT
    Ms = float(row["Mb"]) * MASS_UNIT

    tr = 0

    rhosx = float(row["rhoss"])
    rsx = float(row["rss"])

    rhoss = float(row["rhoss"])
    rss = float(row["rss"])

    rhoH = float(row["rhoH"])
    rH = float(row["rH"])

    c200 = float(row["c200"])
    cvir = float(row["cvir"])

    r = np.asarray(r, dtype=float)


    rho = np.array([rhoDZcore_cal(Mh,Ms,rr,rhosx,c200,rsx,tr,rhoH,rH,rhoss,rss) for rr in r], dtype=float)

    return rho

# ============================================================
# 读取数据
# ============================================================
df = pd.read_csv(CSV_FILE)

df = df.iloc[1:].reset_index(drop=True)

required_columns = ["i", "tr","rhosx", "rsx","rhoss", "rss","rhoH", "rH","cvir", "c200","Mh", "Mb"]

missing_columns = [
    name for name in required_columns
    if name not in df.columns
]

if missing_columns:
    raise ValueError(
        "CSV is missing the following columns: "
        + ", ".join(missing_columns)
    )

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required_columns)

print("Number of subhalos =", len(df))


# ============================================================
# subhalo density profiles
# ============================================================
profiles = []

for _, row in df.iterrows():
    rsx =float(row["rsx"])
    rss = float(row["rss"])
    cvir = float(row["cvir"])
    Rvir = cvir * rss
    r = x_grid * Rvir

    rho = rho_hybrid(row, r)
    rho_original = rho_original_parametric(row, r)
    rho_cdm = rho_CDM(row, r)

    valid = (
    np.isfinite(rho)
    & (rho > 0.0)
    & np.isfinite(rho_original)
    & (rho_original > 0.0)
    & np.isfinite(rho_cdm)
    & (rho_cdm > 0.0)
    & np.isfinite(r)
    & (r > 0.0)
)   
    if rsx>0.5:
        profiles.append({
        "i": int(row["i"]),
        "tr": float(row["tr"]),
        "Rvir": Rvir,
        "rss": rss,
        "r": r[valid],
        "x": x_grid[valid],
        "rho": rho[valid],
        "rho_original": rho_original[valid],
        "rho_cdm": rho_cdm[valid]
    })


# ============================================================
# plot
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.2, 5.6)
)

ax_left = axes[0]
ax_right = axes[1]

cmap = plt.get_cmap("Blues")


# ------------------------------------------------------------
# left
# ------------------------------------------------------------
for profile in profiles:
    
    halo_id = int(profile["i"])

    if halo_id in highlight_ids:
        line_style = "-"
        line_width = 1
        line_alpha = 0.5
    else:
        line_style = ":"
        line_width = 0.33
        line_alpha = 0.23
    
    # CDM
    ax_left.plot(
        profile["x"],
        profile["rho_cdm"],
        color="k",
        linestyle=line_style,
        linewidth=line_width,
        alpha=line_alpha,
        rasterized=True
    )
    
    ax_left.plot(
        profile["x"],
        profile["rho_original"],
        color="b",
        linestyle=line_style,
        linewidth=line_width,
        alpha=line_alpha,
        rasterized=True
    )
    

ax_left.set_xscale("log")
ax_left.set_yscale("log")

ax_left.set_xlabel(
    r"$r/R_{\rm vir}$",
    fontsize=17
)


ax_left.set_ylabel(
    r"$\rho_{\rm DM}\ "
    r"(M_\odot\,{\rm kpc}^{-3})$",
    fontsize=17
)

ax_left.tick_params(
    axis="both",
    which="major",
    labelsize=13
)

ax_left.tick_params(
    axis="both",
    which="minor",
    labelsize=11
)

ax_left.grid(
    which="major",
    linestyle="--",
    alpha=0.25
)


ax_left.set_xlim(
    1e-3,6e-1)
ax_left.set_ylim(
    1e5,4e11
)



# ------------------------------------------------------------
# right
# ------------------------------------------------------------
for profile in profiles:

    halo_id = int(profile["i"])
    
    if halo_id in highlight_ids:
        line_style = "-"
        line_width = 1
        line_alpha = 0.5
    else:
        line_style = ":"
        line_width = 0.33
        line_alpha = 0.23

    # CDM

    ax_right.plot(
        profile["x"],
        profile["rho_cdm"],
        color="k",
        linestyle=line_style,
        linewidth=line_width,
        alpha=line_alpha,
        rasterized=True
    )

    ax_right.plot(
        profile["x"],
        profile["rho"],
        color='r',
        linestyle=line_style,
        linewidth=line_width,
        alpha=line_alpha,
        rasterized=True
    )

   
    
ax_right.set_xscale("log")
ax_right.set_yscale("log")

ax_right.set_xlim(
    1e-3,6e-1
)
ax_right.set_ylim(
    1e5,4e11
)


ax_right.set_xlabel(
    r"$r/R_{\rm vir}$",
    fontsize=17
)

ax_right.set_ylabel(
    r"$\rho_{\rm DM}\ "
    r"(M_\odot\,{\rm kpc}^{-3})$",
    fontsize=17
)

ax_right.tick_params(
    axis="both",
    which="major",
    labelsize=13
)

ax_right.tick_params(
    axis="both",
    which="minor",
    labelsize=11
)

ax_right.grid(
    which="major",
    linestyle="--",
    alpha=0.25
)




# ------------------------------------------------------------
# legend
# ------------------------------------------------------------



legend_elements = [
    Line2D(
        [0], [0],
        color="k",
        linewidth=1.5,
        alpha=0.8,
        label="CDM"
    ),
    
    Line2D(
        [0], [0],
        color="b",
        linestyle="-",
        linewidth=2.0,
        label="PSIDM"
    ) 
]

from matplotlib.lines import Line2D

legend_handles = [
   
    Line2D(
        [0], [0],
        color="k",
        linestyle="-",
        linewidth=2.0,
        label="CDM"
    ),


    Line2D(
        [0], [0],
        color="r",
        linestyle="-",
        linewidth=2.0,
        label="Recalibrated PSIDM"
    ),
    
]

ax_right.legend(
    handles=legend_handles,
    loc="lower left",
    fontsize=12,
    frameon=True
)


ax_left.legend(
    handles=legend_elements,
    fontsize=12,
    loc="lower left",
    frameon=True
)


text_str_left = (
    "Original PSIDM\n"
    r"$\sigma_0/m = 70 \, \mathrm{cm}^2/\mathrm{g}, \,$"
    r"$w = 120 \, \mathrm{km/s}$"
)

ax_left.text(0.35, 0.95, text_str_left, transform=ax_left.transAxes,
        fontsize=12, verticalalignment='top',multialignment='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.5))

text_str_right = (
    "Recalibrated PSIDM\n"
    r"$\sigma_0/m = 70 \, \mathrm{cm}^2/\mathrm{g}, \,$"
    r"$w = 120 \, \mathrm{km/s}$"
)

ax_right.text(0.35, 0.95, text_str_right, transform=ax_right.transAxes,
        fontsize=12, verticalalignment='top',multialignment='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.5))



fig.tight_layout()

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)



plt.show()

print("Saved:", OUTPUT_PDF)

