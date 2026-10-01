"""
Sterile neutrino physics engine.

Self-contained computation of:
  1. Dodelson-Widrow (DW) production  -> delta_N_eff, relic density
  2. Shi-Fuller (SF) resonant production -> delta_N_eff, relic density
  3. BBN impact (parameterised Yp, D/H as functions of N_eff)
  4. Radiative decay rate -> X-ray line predictions
  5. Free-streaming length -> warm DM structure constraints

All formulae are from the literature with references in docstrings.
Units: natural unless noted.  Masses in eV.

References:
  [DW94]  Dodelson & Widrow, PRL 72, 17 (1994)
  [SF99]  Shi & Fuller, PRL 82, 2832 (1999)
  [Pal82] Pal & Wolfenstein, PRD 25, 766 (1982) -- radiative decay
  [PPU18] Pitrou, Coc, Uzan, Vangioni, PhysRep 754, 1 (2018) -- BBN
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Physical constants ──────────────────────────────────────────────
GF = 1.1663788e-5       # Fermi constant [GeV^-2]
MW = 80.379e3            # W mass [MeV]
MZ = 91.1876e3           # Z mass [MeV]
ALPHA_EM = 1.0 / 137.036
T_CMB = 2.7255           # CMB temperature today [K]
T_NU_OVER_T_GAMMA = (4.0 / 11.0) ** (1.0 / 3.0)

# Cosmological parameters (Planck 2018 best-fit)
OMEGA_B_H2 = 0.02237

# Omega h^2 of one thermal neutrino species per eV of mass is 1/93.14
OMEGA_NU_H2_PER_EV_INV = 93.14

# Regime boundaries for the Delta-N_eff-only treatment of BBN
RELATIVISTIC_AT_BBN_MAX_EV = 1e5   # m_s << T_BBN ~ 1 MeV
DECAY_PHYSICS_MIN_EV = 1e6         # >= ~1 MeV: non-relativistic / decaying
                                   # during BBN — injection not modeled


def mass_regime(m_s_eV: float) -> dict:
    """Which physics regime a mass falls in, and whether this server's
    Delta-N_eff-only BBN treatment is valid there."""
    if m_s_eV < 10.0:
        name, valid, note = "eV (short-baseline)", True, (
            "partial thermalisation uses a crude fit; a full QKE "
            "(PRyMordial-nu-sterile) gives Delta N_eff = 0.93 at "
            "Delta m^2 = 1 eV^2, sin^2 2theta = 1e-3 where this fit gives 0.40")
    elif m_s_eV <= RELATIVISTIC_AT_BBN_MAX_EV:
        name, valid, note = "keV (warm dark matter)", True, (
            "relativistic at BBN; Delta N_eff follows from the DW relic density")
    elif m_s_eV < DECAY_PHYSICS_MIN_EV:
        name, valid, note = "100 keV - 1 MeV (transitional)", False, (
            "becoming non-relativistic during BBN; Delta N_eff treatment "
            "is approximate")
    else:
        name, valid, note = "MeV+ (heavy neutral lepton)", False, (
            "OUTSIDE MODEL VALIDITY: non-relativistic and decaying during "
            "BBN. Decay-product injection (photodissociation, hadronic "
            "cascades) and entropy release are not modeled, so BBN numbers "
            "here are not constraints")
    return {"regime": name, "bbn_treatment_valid": valid, "note": note}


# ── Observational data ──────────────────────────────────────────────

@dataclass
class Observation:
    """A measured quantity with Gaussian uncertainty."""
    value: float
    sigma: float
    source: str
    year: int


OBS_DATA = {
    "Yp": Observation(
        value=0.2449, sigma=0.0040,
        source="Aver, Olive & Skillman, JCAP 07 (2021) 029", year=2021,
    ),
    "D_H": Observation(
        value=2.547, sigma=0.025,
        source="Cooke, Pettini & Steidel, ApJ 855 (2018) 102", year=2018,
    ),
    "Neff_CMB": Observation(
        value=2.99, sigma=0.17,
        source="Planck 2018 (TT,TE,EE+lowE+lensing+BAO)", year=2018,
    ),
    "Neff_CMB_S4": Observation(
        value=3.044, sigma=0.03,
        source="CMB-S4 forecast", year=2030,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# 1. DODELSON-WIDROW PRODUCTION
# ═══════════════════════════════════════════════════════════════════

def dw_delta_neff(m_s_eV: float, sin2_2theta: float) -> float:
    """Contribution to N_eff from Dodelson-Widrow production.

    For eV-scale (SBL): can fully thermalise -> delta_Neff ~ 1.
    For keV-scale (DM): non-thermal population, delta_Neff << 1.

    Refs: Abazajian, Fuller & Patel, PRD 64, 023501 (2001)
          Dolgov & Hansen, APP 16, 339 (2002)
          Hannestad, Tamborra & Tram, JCAP 07 (2012) 025
    """
    if sin2_2theta <= 0 or m_s_eV <= 0:
        return 0.0

    # eV-scale: SBL anomaly regime
    if m_s_eV < 10.0:
        if sin2_2theta > 0.01:
            return 1.0  # full thermalisation
        # partial thermalisation: Hannestad+ (2012) Eq. 3.5
        dn = sin2_2theta * 10**2.6
        return min(dn, 1.0)

    # >= 10 eV: non-resonant DW population. Its number density relative to
    # one thermal neutrino species is fixed by the relic density,
    # f = Omega_s h^2 * (93.14 eV / m_s), and since it is still relativistic
    # at BBN (m_s << 1 MeV) it contributes Delta N_eff = f.
    # (The previous keV formula scaled as (1/m_s[eV])^0.5 — a unit-dependent
    # factor — and disagreed with dw_relic_density by 2-65x.)
    f_thermal = dw_relic_density(m_s_eV, sin2_2theta) * OMEGA_NU_H2_PER_EV_INV / m_s_eV
    return min(f_thermal, 1.0)


def dw_relic_density(m_s_eV: float, sin2_2theta: float) -> float:
    """DW relic density Omega_s h^2.

    Omega_s h^2 ~ 0.3 * (sin^2 2theta / 3e-9) * (m_s / 3 keV)^1.8

    Ref: Dodelson & Widrow (1994); Abazajian+ (2001) Eq. 16
    """
    m_s_keV = m_s_eV / 1e3
    if m_s_keV <= 0 or sin2_2theta <= 0:
        return 0.0
    return 0.3 * (sin2_2theta / 3e-9) * (m_s_keV / 3.0) ** 1.8


# ═══════════════════════════════════════════════════════════════════
# 2. SHI-FULLER RESONANT PRODUCTION
# ═══════════════════════════════════════════════════════════════════

def sf_relic_density(
    m_s_eV: float,
    sin2_2theta: float,
    lepton_asymmetry: float = 1e-3,
) -> float:
    """Shi-Fuller resonant production relic density.

    Omega_s h^2 ~ 0.12 * (sin^2 2theta / 2e-13) * (m_s / 7 keV)^2
                       * (L_6 / 10)^0.5

    Ref: Shi & Fuller, PRL 82, 2832 (1999)
    """
    m_s_keV = m_s_eV / 1e3
    L6 = lepton_asymmetry * 1e6
    if m_s_keV <= 0 or sin2_2theta <= 0 or L6 <= 0:
        return 0.0
    return 0.12 * (sin2_2theta / 2e-13) * (m_s_keV / 7.0) ** 2 * (L6 / 10.0) ** 0.5


# ═══════════════════════════════════════════════════════════════════
# 3. BBN IMPACT -- parameterised predictions
# ═══════════════════════════════════════════════════════════════════

# Fits to the full PRyMordial network (63 reactions; Burns, Tait & Valli
# 2023) on a grid Delta N_eff in [-1, 3], omega_b h^2 = 0.02237 x (1 +/- 0.15),
# tau_n = 878.4 +/- 12 s.  Max deviation from PRyMordial: Yp 0.11%, D/H 0.25%,
# Li7/H 0.14%.  Variables: dn = N_eff - 3.044, L = ln(omega_b h^2 / 0.02237),
# dt = tau_n - 878.4 s.  Columns: 1, dn, dn^2, L, L^2, dn*L, dt.
# (Previous coefficients had dYp/dN_eff = 0.0016 — 8x below the network's
# 0.0126 — a wrong-sign tau_n dependence, and a wrong-sign Li7 slope.)
_FIT_YP = (0.246815, 0.013509, -0.000762, 0.009955, 0.002371, -0.000218, 0.000207)
_FIT_LN_DH = (0.893364, 0.135402, -0.007414, -1.63937, -0.105953, 0.016387, 0.000477)
_FIT_LN_LI7 = (1.698117, -0.090997, 0.000958, 2.062926, -0.464125, 0.076803, 0.00042)
BBN_FIT_RANGE = {"neff": (2.044, 6.044), "omega_b_h2": (0.0190, 0.0257),
                 "tau_n": (866.4, 890.4)}


def _bbn_basis(neff, omega_b_h2, tau_n):
    dn = neff - 3.044
    L = np.log(omega_b_h2 / 0.02237)
    dt = tau_n - 878.4
    return (1.0, dn, dn * dn, L, L * L, dn * L, dt)


def _apply(coeffs, basis):
    return sum(c * b for c, b in zip(coeffs, basis))


def bbn_yp(
    neff: float = 3.044,
    omega_b_h2: float = OMEGA_B_H2,
    tau_n: float = 878.4,
) -> float:
    """Primordial He-4 mass fraction Y_p (PRyMordial-calibrated fit).

    SM (N_eff = 3.044, Planck omega_b, tau_n = 878.4 s): 0.2468;
    dYp/dN_eff = 0.0126 near the SM.
    """
    return float(_apply(_FIT_YP, _bbn_basis(neff, omega_b_h2, tau_n)))


def bbn_dh(
    neff: float = 3.044,
    omega_b_h2: float = OMEGA_B_H2,
    tau_n: float = 878.4,
) -> float:
    """Primordial deuterium D/H * 10^5 (PRyMordial-calibrated fit).

    SM: 2.445 (the known ~1.5 sigma D/H tension with 2.547 +/- 0.025);
    d(D/H x 1e5)/dN_eff = 0.33; D/H ~ omega_b^-1.64.
    """
    return float(np.exp(_apply(_FIT_LN_DH, _bbn_basis(neff, omega_b_h2, tau_n))))


def bbn_li7(
    neff: float = 3.044,
    omega_b_h2: float = OMEGA_B_H2,
    tau_n: float = 878.4,
) -> float:
    """Primordial 7Li/H * 10^10 (PRyMordial-calibrated fit).

    SM: 5.47 vs observed ~1.6 (the lithium problem). Li7 DEcreases with
    extra radiation (-0.48 per unit N_eff), so Delta N_eff cannot solve it.
    """
    return float(np.exp(_apply(_FIT_LN_LI7, _bbn_basis(neff, omega_b_h2, tau_n))))


# ═══════════════════════════════════════════════════════════════════
# 4. RADIATIVE DECAY -> X-RAY LINE
# ═══════════════════════════════════════════════════════════════════

def radiative_decay_rate(m_s_eV: float, sin2_2theta: float) -> float:
    """Gamma(nu_s -> nu_a + gamma) in s^-1.

    Gamma = 9 alpha G_F^2 sin^2(2theta) m_s^5 / (1024 pi^4)
          = 1.361e-32 s^-1 * (sin^2 2theta / 1e-10) * (m_s / keV)^5
          (equivalently 1.36e-29 s^-1 for sin^2 2theta = 1e-7)

    Ref: Pal & Wolfenstein, PRD 25, 766 (1982); Boyarsky+ (2009).
    (Previously normalised to 1e-10 with the 1e-7 coefficient: 1014x too
    fast, i.e. lifetimes 1014x too short and X-ray fluxes 1014x too high.)
    """
    m_s_keV = m_s_eV / 1e3
    if m_s_keV <= 0 or sin2_2theta <= 0:
        return 0.0
    return 1.361e-32 * (sin2_2theta / 1e-10) * m_s_keV ** 5


def radiative_lifetime(m_s_eV: float, sin2_2theta: float) -> float:
    """Radiative lifetime tau = 1/Gamma in seconds."""
    rate = radiative_decay_rate(m_s_eV, sin2_2theta)
    if rate <= 0:
        return np.inf
    return 1.0 / rate


def xray_line_energy_keV(m_s_eV: float) -> float:
    """X-ray photon energy E_gamma = m_s / 2 in keV."""
    return m_s_eV / 2e3


def xray_flux_from_halo(
    m_s_eV: float,
    sin2_2theta: float,
    M_halo_msun: float = 1e14,
    d_Mpc: float = 100.0,
    f_dm: float = 1.0,
) -> float:
    """X-ray line flux from sterile nu DM decay in a halo.

    F = Gamma * f_DM * M_halo / (4 pi d^2 m_s)  [photons/s/cm^2]
    """
    rate = radiative_decay_rate(m_s_eV, sin2_2theta)
    MSUN_EV = 1.116e66
    MPC_CM = 3.0857e24
    M_eV = M_halo_msun * MSUN_EV
    d_cm = d_Mpc * MPC_CM
    N_sterile = f_dm * M_eV / m_s_eV
    return rate * N_sterile / (4.0 * np.pi * d_cm ** 2)


# ═══════════════════════════════════════════════════════════════════
# 5. STRUCTURE FORMATION
# ═══════════════════════════════════════════════════════════════════

def free_streaming_length_kpc(m_s_eV: float, mechanism: str = "DW") -> float:
    """Free-streaming length lambda_fs in kpc.

    lambda_fs ~ 500 kpc * (1 keV / m_s) * (106.75 / g_s)^{1/3}

    Ref: Boyarsky, Ruchayskiy & Shaposhnikov (2009) Eq. 15
    """
    m_s_keV = m_s_eV / 1e3
    if m_s_keV <= 0:
        return np.inf
    g_s = 10.75 if mechanism == "DW" else 30.0
    return 500.0 * (1.0 / m_s_keV) * (106.75 / g_s) ** (1.0 / 3.0)


# ═══════════════════════════════════════════════════════════════════
# 6. CHI-SQUARED & CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════

# Delta chi^2 threshold for excluding a sterile neutrino at 95% CL: its only
# BBN effect here is one number (Delta N_eff), so 1 degree of freedom.
DELTA_CHI2_95 = 3.84

# Theory (nuclear-rate) errors as quoted by PRyMordial
THEORY_SIGMA = {"Yp": 0.0003, "D_H": 0.06}


def _chi2_parts(neff, omega_b_h2, tau_n):
    yp_obs, dh_obs = OBS_DATA["Yp"], OBS_DATA["D_H"]
    sig_yp = np.sqrt(yp_obs.sigma ** 2 + THEORY_SIGMA["Yp"] ** 2)
    sig_dh = np.sqrt(dh_obs.sigma ** 2 + THEORY_SIGMA["D_H"] ** 2)
    yp_pred = bbn_yp(neff, omega_b_h2, tau_n)
    dh_pred = bbn_dh(neff, omega_b_h2, tau_n)
    c2_yp = ((yp_pred - yp_obs.value) / sig_yp) ** 2
    c2_dh = ((dh_pred - dh_obs.value) / sig_dh) ** 2
    return yp_pred, dh_pred, sig_yp, sig_dh, c2_yp, c2_dh


def chi2_bbn(neff: float, omega_b_h2: float = OMEGA_B_H2,
             tau_n: float = 878.4) -> dict:
    """Chi-squared of BBN predictions vs observed Yp and D/H.

    `delta_chi2_vs_sm` (relative to N_eff = 3.044 at the same omega_b and
    tau_n) is the quantity to exclude on: the SM itself carries chi2 ~ 2.7
    from the D/H tension, which is not evidence about sterile neutrinos.
    """
    yp_pred, dh_pred, sig_yp, sig_dh, c2_yp, c2_dh = _chi2_parts(neff, omega_b_h2, tau_n)
    yp_obs, dh_obs = OBS_DATA["Yp"], OBS_DATA["D_H"]
    c2 = c2_yp + c2_dh
    *_, sm_yp, sm_dh = _chi2_parts(3.044, omega_b_h2, tau_n)
    d_c2 = c2 - (sm_yp + sm_dh)

    return {
        "Yp_predicted": round(yp_pred, 5),
        "Yp_observed": yp_obs.value,
        "Yp_sigma": round(sig_yp, 5),
        "chi2_Yp": round(c2_yp, 4),
        "DH5_predicted": round(dh_pred, 4),
        "DH5_observed": dh_obs.value,
        "DH5_sigma": round(sig_dh, 4),
        "chi2_DH": round(c2_dh, 4),
        "chi2_total": round(c2, 4),
        "delta_chi2_vs_sm": round(d_c2, 4),
        "neff": neff,
        "excluded_95": bool(d_c2 > DELTA_CHI2_95),
        "consistent_2sigma": bool(d_c2 <= DELTA_CHI2_95),
    }


# ═══════════════════════════════════════════════════════════════════
# 7. PARAMETER-SPACE SCANNER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class XrayLimit:
    """Upper limit on sin^2(2theta) from X-ray observations."""
    m_s_keV: float
    sin2_2theta_upper: float
    source: str


XRAY_LIMITS = [
    XrayLimit(3.0,  5.0e-10, "NuSTAR M31 (Roach+ 2020)"),
    XrayLimit(5.0,  2.5e-11, "NuSTAR M31 (Roach+ 2020)"),
    XrayLimit(7.0,  5.0e-12, "NuSTAR M31 (Roach+ 2020)"),
    XrayLimit(10.0, 2.0e-12, "NuSTAR M31 (Roach+ 2020)"),
    XrayLimit(15.0, 5.0e-13, "NuSTAR M31 (Roach+ 2020)"),
    XrayLimit(20.0, 2.0e-13, "NuSTAR M31 (Roach+ 2020)"),
    XrayLimit(3.5,  3.0e-10, "XMM-Newton blank sky (Dessert+ 2020)"),
    XrayLimit(7.1,  7.0e-12, "XMM-Newton blank sky (Dessert+ 2020)"),
    XrayLimit(10.0, 2.5e-12, "XMM-Newton blank sky (Dessert+ 2020)"),
    XrayLimit(15.0, 6.0e-13, "XMM-Newton blank sky (Dessert+ 2020)"),
]

LYMAN_ALPHA_LOWER_MASS_KEV = 5.3

LINE_35_KEV = {
    "E_keV": 3.55,
    "m_s_eV": 7.1e3,
    "sin2_2theta_best": 7e-11,
    "sin2_2theta_range": (2e-11, 2e-10),
    "status": "debated",
    "refs": [
        "Bulbul+ (2014), ApJ 789, 13",
        "Boyarsky+ (2014), PRL 113, 251301",
        "Dessert+ (2020), Science 367, 1465 (exclusion)",
    ],
}

SBL_BEST = {
    "delta_m2_eV2": 1.3,
    "sin2_2theta": 0.42,
    "source": "BEST experiment, PRL 128, 232501 (2022)",
    "status": "in tension with STEREO, PROSPECT, Daya Bay nulls",
}


def scan_parameter_space(
    m_range_eV: tuple[float, float] = (1e2, 1e5),
    theta_range: tuple[float, float] = (1e-15, 1e-5),
    n_m: int = 80,
    n_theta: int = 80,
    mechanism: str = "DW",
) -> dict:
    """Scan the (m_s, sin^2 2theta) plane and evaluate constraints.

    Returns arrays suitable for contour plotting.
    """
    m_arr = np.geomspace(m_range_eV[0], m_range_eV[1], n_m)
    t_arr = np.geomspace(theta_range[0], theta_range[1], n_theta)

    neff_grid = np.zeros((n_m, n_theta))
    omega_grid = np.zeros((n_m, n_theta))
    chi2_grid = np.zeros((n_m, n_theta))
    dchi2_grid = np.zeros((n_m, n_theta))
    tau_grid = np.zeros((n_m, n_theta))

    for i, ms in enumerate(m_arr):
        for j, s22t in enumerate(t_arr):
            if mechanism == "DW":
                dn = dw_delta_neff(ms, s22t)
                om = dw_relic_density(ms, s22t)
            else:
                dn = dw_delta_neff(ms, s22t) * 0.1
                om = sf_relic_density(ms, s22t)
            neff_grid[i, j] = 3.044 + dn
            omega_grid[i, j] = om
            c = chi2_bbn(3.044 + dn)
            chi2_grid[i, j] = c["chi2_total"]
            dchi2_grid[i, j] = c["delta_chi2_vs_sm"]
            tau_grid[i, j] = radiative_lifetime(ms, s22t)

    return {
        "m_s_eV": m_arr,
        "sin2_2theta": t_arr,
        "neff": neff_grid,
        "omega_h2": omega_grid,
        "chi2_bbn": chi2_grid,
        "delta_chi2_bbn": dchi2_grid,
        "lifetime_s": tau_grid,
    }
