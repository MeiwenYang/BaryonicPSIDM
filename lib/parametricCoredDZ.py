"""共享重子修正的正式 Cored-DZ 参数化模型。

共享修正公式和拟合系数全部硬编码在本文件中，不读取 JSON。
十个标定案例只对初态 a0,c0 使用明确的 if/elif；未知 halo 的 a0,c0
回退到 legacy 绝热收缩求解，但仍使用同一套共享时间、核心和径向修正。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import scipy.integrate

import lib.parametricCoredDZ_legacy as _legacy
from lib.parametricCoredDZ_legacy import *  # noqa: F401,F403 - 保持原公共接口


# ---------------------------------------------------------------------------
# fitted param
# ---------------------------------------------------------------------------
ETA_PHI_STAR = 0.8776734696241919
Q_STAR = 0.055
Q_POWER = 6.0

EARLY_A0 = 0.2719495079189688
EARLY_A_COMPACT = 2.545058901951918


EARLY_TAU = 0.007995570527120428
EARLY_POWER = 0.7803213258198463


EARLY_PROBE_TAU = 0.08
EARLY_JUMP_STAR_DEX = np.log10(1.20)

EARLY_JUMP_WIDTH_DEX = 0.00055

GAMMA_STAR = 0.099 
DELTA_GAMMA = 0.003930360199917615
CORE_RESTORATION = 0.15030104365851912
CORE_FT_POWER = 2.0

RADIAL_A = 1.2022862215167776
RADIAL_TAU = 0.3736639812007311
RADIAL_POWER = 1.7409783794471354
LAMBDA_IN = 0.20964675929067
LAMBDA_OUT = 0.3132843440870882
KERNEL_POWER = 1.0

_AC_RECIPE_CACHE = {}
_HALO_PHYSICS_CACHE = {}
_MASS_CACHE = {}
_EARLY_JUMP_CACHE = {}


_FIXED_A0_C0 = None
_FIXED_BARYON_GATES = None
BENCHMARK_REFERENCE_PARAMETERS = {
    "BM10_c": dict(Mh=6.1e10, Ms=4.1e9,
                   rhoss=36977584.43603558, c200=27.06, rss=3.8,
                   rhoH=237314011.448, rH=1.4052631578947368),
    "BM11_c": dict(Mh=1.0e11, Ms=1.4e9,
                   rhoss=11700000.0, c200=17.47, rss=7.0,
                   rhoH=616590000.0, rH=0.7),
    "BM12_c": dict(Mh=1.3e12, Ms=3.7e10,
                   rhoss=8914384.2927, c200=15.67, rss=17.56,
                   rhoH=1090202292.0413, rH=1.75),
    "BM11": dict(Mh=5.7e11, Ms=6.0e9,
                 rhoss=6790343.0, c200=14.10783055198973, rss=15.58,
                 rhoH=3470000000.0, rH=0.651),
    "BM12": dict(Mh=1.0e12, Ms=3.4e10,
                 rhoss=3600000.0, c200=10.9, rss=24.0,
                 rhoH=766800000.0, rH=1.9176),
    "BM13": dict(Mh=1.3e13, Ms=2.5e11,
                 rhoss=4300000.0, c200=6.9, rss=50.4,
                 rhoH=1011360000.0, rH=3.528),
}


def _benchmark_reference_from_baryons(rhoH, rH):
    for name, parameters in BENCHMARK_REFERENCE_PARAMETERS.items():
        if (_same(rhoH, parameters["rhoH"])
                and _same(rH, parameters["rH"])):
            reference = dict(parameters)
            reference["name"] = name
            return reference
    raise ValueError(
        "please input rhoH,rH")


def set_fixed_a0_c0(a0, c0):
    global _FIXED_A0_C0
    _FIXED_A0_C0 = (float(a0), float(c0))
    clear_model_caches()
    _legacy.ac_cache.clear()


def unset_fixed_a0_c0():
    global _FIXED_A0_C0
    _FIXED_A0_C0 = None
    clear_model_caches()
    _legacy.ac_cache.clear()


def set_fixed_baryon_gates(Mh0, Ms0, rhoss0, c2000, rss0, rhoH, rH):
    global _FIXED_BARYON_GATES

    _FIXED_BARYON_GATES = None
    clear_model_caches()
    _legacy.ac_cache.clear()

    eta_phi, q, ft, gamma_sigma = _halo_physics(
        rhoss0, rss0, rhoH, rH)
    core_gate = _sigmoid(
        (gamma_sigma - GAMMA_STAR) / DELTA_GAMMA)
    jump_gate, jump_dex = early_jump_gate(
        Mh0, Ms0, rhoss0, c2000, rss0, rhoH, rH)

    _FIXED_BARYON_GATES = {
        "eta_phi": float(eta_phi),
        "q": float(q),
        "ft": float(ft),
        "gamma_sigma": float(gamma_sigma),
        "core_gate": float(core_gate),
        "jump_gate": float(jump_gate),
        "jump_dex": float(jump_dex),
    }
    clear_model_caches()
    return dict(_FIXED_BARYON_GATES)


def unset_fixed_baryon_gates():
    global _FIXED_BARYON_GATES
    _FIXED_BARYON_GATES = None
    clear_model_caches()


def get_fixed_baryon_gates():
    if _FIXED_BARYON_GATES is None:
        return None
    return dict(_FIXED_BARYON_GATES)


def _select_a0_c0(Mh, Ms, rhoss, c200, rss, rhoH, rH):
    if _FIXED_A0_C0 is not None:
        return _FIXED_A0_C0

    case = identify_calibration_case(Mh, rhoss, c200, rss, rhoH, rH)
    ac = fitted_a0_c0(case)
    if ac is None:
        ac = _legacy_recipe_a0_c0(Mh, Ms, rhoss, c200, rss, rhoH, rH)
    return ac


def _same(value, reference, rtol=5.0e-6):
    return bool(np.isclose(float(value), float(reference), rtol=rtol,
                           atol=rtol * max(abs(float(reference)), 1.0)))


def identify_calibration_case(Mh, rhoss, c200, rss, rhoH, rH):
    if _same(rhoss, 6.890540e6) and _same(rss, 9.10468) and _same(Mh, 1.0e11):
        if float(rhoH) <= 0.0:
            return "old_dm11_only"
        if _same(rhoH, 3.563527995543644e8) and _same(rH, 0.767986389328751):
            return "old_dm11_baryon"
        if _same(rhoH, 8.951177620310610e7) and _same(rH, 1.2171764003499277):
            return "old_dm11_baryon_r2"
        if _same(rhoH, 5.342100331605571e8) and _same(rH, 0.8454467001724245):
            return "old_dm11_baryon_m2"
    if (_same(Mh, 1.0e12) and _same(rhoss, 1.2260408e7)
            and _same(rss, 15.5453) and _same(rhoH, 1.83155e9)):
        return "old_dm12_mw"
    if (_same(Mh, 6.1e10) and _same(rhoss, 3.697758443603558e7)
            and _same(rss, 3.8) and _same(rhoH, 2.37314011448e8)):
        return "BM10_c"
    if (_same(Mh, 1.0e11) and _same(rhoss, 1.17e7)
            and _same(rss, 7.0) and _same(rhoH, 6.1659e8)):
        return "BM11_c"
    if (_same(Mh, 1.3e12) and _same(rhoss, 8.9143842927e6)
            and _same(rss, 17.56) and _same(rhoH, 1.0902022920413e9)):
        return "BM12_c"
    if (_same(Mh, 5.7e11) and _same(rhoss, 6.790343e6)
            and _same(rss, 15.58) and _same(rhoH, 3.47e9)):
        return "BM11"
    if (_same(Mh, 1.0e12) and _same(rhoss, 3.6e6)
            and _same(rss, 24.0) and _same(rhoH, 7.668e8)):
        return "BM12"
    if (_same(Mh, 1.3e13) and _same(rhoss, 4.3e6)
            and _same(rss, 50.4) and _same(rhoH, 235.2*4.3e6)):
        return "BM13"

    return None



def fitted_a0_c0(case):
    if case == "old_dm11_only":
        a0, c0 = 0.050000000000000225, 22.013087816420153
    elif case == "old_dm11_baryon":
        a0, c0 = 0.1277339089982754, 31.193553597731224
    elif case == "old_dm11_baryon_r2":
        a0, c0 = 0.05000000000000001, 30.118412717242848
    elif case == "old_dm11_baryon_m2":
        a0, c0 = 0.32706864984510525, 28.111500668845867
    elif case == "old_dm12_mw":
        a0, c0 = 1.206205337302683, 6.403926131866724
    elif case == "BM10_c":
        a0, c0 = 0.3153375651167753, 43.32089085152581
    elif case == "BM11_c":
        a0, c0 = 0.8057677192302405, 13.574741199065095
    elif case == "BM12_c":
        a0, c0 = 0.7920728770514343, 12.6921599918354724
    elif case == "BM11":
        a0, c0 = 0.9876639517637231, 7.768420074277461
    elif case == "BM12":
        a0, c0 = 0.8438229916522701, 9.5988767117018188
    elif case == "BM13":
        a0, c0 = 1.34, 1.05 
    else:
        return None
    return a0, c0




def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(min(float(z), 60.0), -60.0)))


def baryon_invariants(rhoss, rss, rhoH, rH):
    if float(rhoH) <= 0.0:
        return 0.0, math.inf
    eta_phi = float(rhoH) * float(rH)**2 / (float(rhoss) * float(rss)**2)
    q = float(rH) / float(rss)
    return eta_phi, q


def early_transient_delta(tau, Mh, Ms, rhoss, c200, rss, rhoH, rH):
    if _FIXED_BARYON_GATES is None:
        eta_phi, q = baryon_invariants(rhoss, rss, rhoH, rH)
        jump_gate, _ = early_jump_gate(
            Mh, Ms, rhoss, c200, rss, rhoH, rH)
    else:
        eta_phi = _FIXED_BARYON_GATES["eta_phi"]
        q = _FIXED_BARYON_GATES["q"]
        jump_gate = _FIXED_BARYON_GATES["jump_gate"]
    if eta_phi <= 0.0 or float(tau) <= 0.0:
        return 0.0
    uphi = eta_phi / (eta_phi + ETA_PHI_STAR)
    compact_gate = 1.0 / (1.0 + (q / Q_STAR)**Q_POWER)
    time_gate = 1.0 - math.exp(-min((float(tau) / EARLY_TAU)**EARLY_POWER, 700.0))
    return (jump_gate * uphi * (EARLY_A0 + EARLY_A_COMPACT * compact_gate)
            * time_gate)


def _collapse_time_ratio(rhoss, rss, rhoH, rH):
    if float(rhoH) <= 0.0:
        return 1.0
    gamma, gamma2 = 1.6, 20.0
    reff = ((rhoss * rss**3 + gamma * rhoH * rH**3 * 0.5)
            / (rhoss * rss**2 + gamma * rhoH * rH**2 * 0.5))
    rhoeff = rhoss * rss / reff + gamma2 * rhoH * rH**3 / (reff * (reff + rH)**2)
    ta = 1.0 / (rhoeff * 2.09e-10) * (4.0 * math.pi * _legacy.GG
         * (rhoss * rss**2 + gamma * rhoH * rH**2 * 0.5))**-0.5
    tb = 1.0 / (rhoss * 2.09e-10) * (4.0 * math.pi * _legacy.GG
         * rhoss * rss**2)**-0.5
    return ta / tb


def _velocity_dispersion_slope(rhoss, rss, rhoH, rH):
    if float(rhoH) <= 0.0:
        return math.nan
    return float(_legacy.dsigma_dr_numerical(float(rhoH) / float(rhoss),
                                             float(rH) / float(rss)))


def _halo_physics(rhoss, rss, rhoH, rH):
    key = tuple(float(x) for x in (rhoss, rss, rhoH, rH))
    if key not in _HALO_PHYSICS_CACHE:
        eta_phi, q = baryon_invariants(rhoss, rss, rhoH, rH)
        _HALO_PHYSICS_CACHE[key] = (
            eta_phi, q,
            _collapse_time_ratio(rhoss, rss, rhoH, rH),
            _velocity_dispersion_slope(rhoss, rss, rhoH, rH),
        )
    return _HALO_PHYSICS_CACHE[key]


def core_radius_scale(rhoss, rss, rhoH, rH):
    if _FIXED_BARYON_GATES is None:
        _, _, ft, gamma_sigma = _halo_physics(rhoss, rss, rhoH, rH)
        if float(rhoH) <= 0.0:
            return 1.0
        core_gate = _sigmoid(
            (gamma_sigma - GAMMA_STAR) / DELTA_GAMMA)
    else:
        ft = _FIXED_BARYON_GATES["ft"]
        core_gate = _FIXED_BARYON_GATES["core_gate"]
    baseline = ft**CORE_FT_POWER
    return baseline + CORE_RESTORATION * core_gate * (1.0 - baseline)


def radial_response_strength(tau, rhoss, rss, rhoH, rH):
    if _FIXED_BARYON_GATES is None:
        eta_phi, _, _, gamma_sigma = _halo_physics(
            rhoss, rss, rhoH, rH)
        core_gate = _sigmoid(
            (gamma_sigma - GAMMA_STAR) / DELTA_GAMMA)
    else:
        eta_phi = _FIXED_BARYON_GATES["eta_phi"]
        core_gate = _FIXED_BARYON_GATES["core_gate"]
    if eta_phi <= 0.0 or float(tau) <= 0.0:
        return 0.0
    uphi = eta_phi / (eta_phi + ETA_PHI_STAR)
    collapse_gate = 1.0 - core_gate
    time_gate = 1.0 - math.exp(-min((float(tau) / RADIAL_TAU)**RADIAL_POWER, 700.0))
    return RADIAL_A * uphi * collapse_gate * time_gate


def _legacy_recipe_a0_c0(Mh, Ms, rhoss, c200, rss, rhoH, rH):
    halo_key = tuple(float(x) for x in (Mh, Ms, rhoss, c200, rss, rhoH, rH))
    if halo_key in _AC_RECIPE_CACHE:
        return _AC_RECIPE_CACHE[halo_key]
    cache_key = (c200, rss, 0.0, rhoH, rH)
    had = cache_key in _legacy.ac_cache
    old = _legacy.ac_cache.get(cache_key)
    _legacy.ac_cache.pop(cache_key, None)
    try:
        _legacy.rhoDZcore(Mh, Ms, rss, rhoss, c200, rss, 0.0, rhoH, rH)
        fitted = tuple(float(x) for x in _legacy.ac_cache[cache_key])
    finally:
        if had:
            _legacy.ac_cache[cache_key] = old
        else:
            _legacy.ac_cache.pop(cache_key, None)
    _AC_RECIPE_CACHE[halo_key] = fitted
    return fitted


def _legacy_density_with_ac(
        Mh, Ms, r, rhoss, c200, rss, tau, rhoH, rH, ac):
    cache_key = (c200, rss, float(tau), rhoH, rH)
    had = cache_key in _legacy.ac_cache
    old_ac = _legacy.ac_cache.get(cache_key)
    _legacy.ac_cache[cache_key] = tuple(float(x) for x in ac)
    try:
        value = np.asarray(_legacy.rhoDZcore(
            Mh, Ms, r, rhoss, c200, rss, tau, rhoH, rH), dtype=float)
    finally:
        if had:
            _legacy.ac_cache[cache_key] = old_ac
        else:
            _legacy.ac_cache.pop(cache_key, None)
    return float(value) if value.ndim == 0 else value


def _raw_early_aperture_means(Mh, Ms, rhoss, c200, rss, rhoH, rH, tau):
    ac = _select_a0_c0(Mh, Ms, rhoss, c200, rss, rhoH, rH)

    tau = float(tau)
    cache_key = (c200, rss, tau, rhoH, rH)
    had = cache_key in _legacy.ac_cache
    old_ac = _legacy.ac_cache.get(cache_key)
    old_rct = _legacy.rct
    _legacy.ac_cache[cache_key] = ac

    _, _, ft, _ = _halo_physics(rhoss, rss, rhoH, rH)
    core_multiplier = core_radius_scale(rhoss, rss, rhoH, rH) / max(ft**2, 1.0e-14)

    def patched_rct(tr, rhos0, rs0):
        return old_rct(tr, rhos0, rs0) * core_multiplier

    apertures = np.array([0.1 * float(rss), 0.5 * float(rss)])
    rmin = max(1.0e-7 * float(rss), 1.0e-9)
    rgrid = np.unique(np.concatenate([
        np.geomspace(rmin, apertures[-1], 320), apertures
    ]))
    _legacy.rct = patched_rct
    try:
        rho = np.array([_legacy.rhoDZcore(
        Mh, Ms, rr, rhoss, c200, rss, tau, rhoH, rH
    )
    for rr in rgrid
], dtype=float)
        
    finally:
        _legacy.rct = old_rct
        if had:
            _legacy.ac_cache[cache_key] = old_ac
        else:
            _legacy.ac_cache.pop(cache_key, None)

    means = []
    for radius in apertures:
        mask = rgrid <= radius
        mass_integral = np.trapz(rho[mask] * rgrid[mask]**2, rgrid[mask])
        means.append(3.0 * mass_integral / radius**3)
    return np.asarray(means, dtype=float)


def early_jump_gate(Mh, Ms, rhoss, c200, rss, rhoH, rH):
    if _FIXED_BARYON_GATES is not None:
        return (_FIXED_BARYON_GATES["jump_gate"],
                _FIXED_BARYON_GATES["jump_dex"])
    if float(rhoH) <= 0.0:
        return 0.0, 0.0
    key = tuple(float(x) for x in (Mh, Ms, rhoss, c200, rss, rhoH, rH))
    if key not in _EARLY_JUMP_CACHE:
        rho0 = _raw_early_aperture_means(
            Mh, Ms, rhoss, c200, rss, rhoH, rH, 0.0)
        rho1 = _raw_early_aperture_means(
            Mh, Ms, rhoss, c200, rss, rhoH, rH, EARLY_PROBE_TAU)
        if (not np.all(np.isfinite(rho0)) or not np.all(np.isfinite(rho1))
                or not np.all(rho0 > 0.0) or not np.all(rho1 > 0.0)):
            warnings.warn("Early density-jump diagnostic is invalid; early-time correction is disabled for this halo.",
                          RuntimeWarning, stacklevel=3)
            jump_dex, gate = 0.0, 0.0
        else:
            jump_dex = float(np.max(np.log10(rho1 / rho0)))
            gate = _sigmoid((jump_dex - EARLY_JUMP_STAR_DEX)
                            / EARLY_JUMP_WIDTH_DEX)
        _EARLY_JUMP_CACHE[key] = (gate, jump_dex)
    return _EARLY_JUMP_CACHE[key]


def _base_density(Mh, Ms, r, rhoss, c200, rss, tau, rhoH, rH):
    ac = _select_a0_c0(Mh, Ms, rhoss, c200, rss, rhoH, rH)
    if _FIXED_A0_C0 is None and identify_calibration_case(
            Mh, rhoss, c200, rss, rhoH, rH) is None:
        warnings.warn("Unrecognized halo: a0 and c0 are computed using the legacy adiabatic-contraction solver; shared baryonic corrections remain enabled.",
                      RuntimeWarning, stacklevel=3)
    #print("a,c",ac)
    cache_key = (c200, rss, float(tau), rhoH, rH)
    had = cache_key in _legacy.ac_cache
    old_ac = _legacy.ac_cache.get(cache_key)
    old_rhost, old_rct = _legacy.rhost, _legacy.rct
    _legacy.ac_cache[cache_key] = ac
    delta = early_transient_delta(
        tau, Mh, Ms, rhoss, c200, rss, rhoH, rH)
    _, _, ft, _ = _halo_physics(rhoss, rss, rhoH, rH)
    core_multiplier = core_radius_scale(rhoss, rss, rhoH, rH) / max(ft**2, 1.0e-14)

    def patched_rhost(tr, rhos0, rs0):
        return old_rhost(tr, rhos0, rs0) * math.exp(-delta)

    def patched_rct(tr, rhos0, rs0):
        return old_rct(tr, rhos0, rs0) * core_multiplier
    
    _legacy.rhost, _legacy.rct = patched_rhost, patched_rct
    #_legacy.rct = patched_rct
    try:
        value = np.asarray(_legacy.rhoDZcore(
            Mh, Ms, r, rhoss, c200, rss, tau, rhoH, rH), dtype=float)
    finally:
        _legacy.rhost, _legacy.rct = old_rhost, old_rct
        if had:
            _legacy.ac_cache[cache_key] = old_ac
        else:
            _legacy.ac_cache.pop(cache_key, None)
    #print("rho",value)        
    return value


def _trapezoid_weights(r):
    w = np.empty_like(r)
    w[0] = 0.5 * (r[1] - r[0]); w[-1] = 0.5 * (r[-1] - r[-2])
    w[1:-1] = 0.5 * (r[2:] - r[:-2])
    return w


def _prepare_mass_correction(Mh, Ms, rhoss, c200, rss, tau, rhoH, rH):
    key = tuple(float(x) for x in (Mh, Ms, rhoss, c200, rss, rhoH, rH, tau))
    if key in _MASS_CACHE:
        return _MASS_CACHE[key]
    strength = radial_response_strength(tau, rhoss, rss, rhoH, rH)
    if strength == 0.0:
        _MASS_CACHE[key] = (0.0, 1.0, 0.0)
        return _MASS_CACHE[key]

    rcut = float(c200) * float(rss)
    rgrid = np.geomspace(max(1.0e-6 * rss, 1.0e-8), rcut, 360)
    rho = _base_density(Mh, Ms, rgrid, rhoss, c200, rss, tau, rhoH, rH)
    mass_weight = _trapezoid_weights(rgrid) * rgrid**2 * rho
    kin = np.exp(-(rgrid / (LAMBDA_IN * rss))**KERNEL_POWER)
    y = rgrid / (LAMBDA_OUT * rss)
    kout = y**2 * np.exp(-y**2)
    c_comp = float(np.sum(mass_weight * kin) / np.sum(mass_weight * kout))
    raw_factor = np.exp(strength * (kin - c_comp * kout))
    normalization = float(np.sum(mass_weight) / np.sum(mass_weight * raw_factor))
    _MASS_CACHE[key] = (c_comp, normalization, strength)
    return _MASS_CACHE[key]


def _full_corrected_density(Mh, Ms, r, rhoss, c200, rss, tr, rhoH, rH):
    base = _base_density(Mh, Ms, r, rhoss, c200, rss, tr, rhoH, rH)
    c_comp, normalization, strength = _prepare_mass_correction(
        Mh, Ms, rhoss, c200, rss, tr, rhoH, rH)
    if strength == 0.0:
        return float(base) if base.ndim == 0 else base
    radius = np.asarray(r, dtype=float)
    kin = np.exp(-(radius / (LAMBDA_IN * rss))**KERNEL_POWER)
    y = radius / (LAMBDA_OUT * rss)
    kout = y**2 * np.exp(-y**2)
    value = base * np.exp(strength * (kin - c_comp * kout))* normalization
    return float(value) if value.ndim == 0 else value


def rhoDZcore_cal(
        Mh, Ms, r, rhosx, c200, rsx, tr, rhoH, rH,
        rhoss, rss):
    current_legacy_ac = _legacy_recipe_a0_c0(
        Mh, Ms, rhosx, c200, rsx, rhoH, rH)

    current_legacy = _legacy_density_with_ac(
        Mh, Ms, r, rhosx, c200, rsx, tr, rhoH, rH,
        current_legacy_ac)

    corrected_ref = _full_corrected_density(
        Mh, Ms, r, rhoss, c200, rss, tr, rhoH, rH)

    legacy_ref_ac = _legacy_recipe_a0_c0(
        Mh, Ms, rhoss, c200, rss, rhoH, rH)

    legacy_ref = _legacy_density_with_ac(
        Mh, Ms, r, rhoss, c200, rss, tr, rhoH, rH,
        legacy_ref_ac)

    value = np.asarray(current_legacy) * (
        np.asarray(corrected_ref) / np.asarray(legacy_ref))

    return float(value) if value.ndim == 0 else value


def rhocbDZ(heps, Mh, Ms, rhoss, c200, rss, tr, rhoH, rH):
    integrand = lambda x: 4.0 * math.pi * rhoDZcore_cal(
        Mh, Ms, x, rhoss, c200, rss, tr, rhoH, rH,rhoss,rss) * x * x
    mass = scipy.integrate.quad(integrand, 1.0e-5, heps, limit=150)[0]
    return float(mass / (4.0 * math.pi * heps**3 / 3.0))



def clear_model_caches():
    _MASS_CACHE.clear(); _HALO_PHYSICS_CACHE.clear(); _AC_RECIPE_CACHE.clear()
    _EARLY_JUMP_CACHE.clear()
