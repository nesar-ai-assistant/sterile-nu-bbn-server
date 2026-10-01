"""
MCP tool functions for the sterile-neutrino-BBN bridge server.

Every tool returns an ArtifactResult dict:
    {status, files, message, metadata}
Arrays pass between tools as file paths, never through the context window.

Scope: this server predicts what a sterile neutrino PRODUCES (Delta N_eff,
relic density, X-ray decay line, free streaming) and the BBN/X-ray
constraints that follow. Light-element abundances come from a fit to the
PRyMordial network; for a full network calculation at a given N_eff use a
dedicated BBN abundance server (e.g. compute_bbn_abundances / scan_bbn_neff
on the bbn server, if connected).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from tools.physics import (
    BBN_FIT_RANGE,
    DECAY_PHYSICS_MIN_EV,
    DELTA_CHI2_95,
    OBS_DATA,
    SBL_BEST,
    LINE_35_KEV,
    XRAY_LIMITS,
    LYMAN_ALPHA_LOWER_MASS_KEV,
    bbn_dh,
    bbn_li7,
    bbn_yp,
    chi2_bbn,
    dw_delta_neff,
    dw_relic_density,
    free_streaming_length_kpc,
    mass_regime,
    radiative_decay_rate,
    radiative_lifetime,
    scan_parameter_space,
    sf_relic_density,
    xray_flux_from_halo,
    xray_line_energy_keV,
)

DEFAULT_OUTPUT = Path("output")


def _ensure_dir(d: Path) -> Path:
    # Absolute so clients on the same machine (e.g. the HEP-Genesis app) can
    # find returned files regardless of this server's cwd — a relative path
    # in `files` is unreadable from another process.
    d = d.expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _output_dir(output_dir: str) -> Path:
    """The caller's output_dir; the default falls back to $MCP_OUTPUT_DIR
    (set by MCP clients that route outputs per project) before ./output."""
    if output_dir == str(DEFAULT_OUTPUT):
        return Path(os.environ.get("MCP_OUTPUT_DIR", output_dir))
    return Path(output_dir)


def _slug(params: dict) -> str:
    blob = ",".join(f"{k}={params[k]}" for k in sorted(params))
    return hashlib.sha1(blob.encode()).hexdigest()[:6]


XRAY_TARGETS = {
    "Perseus": {"M_msun": 6.65e14, "d_Mpc": 77.7},
    "M31": {"M_msun": 1.3e12, "d_Mpc": 0.78},
    "Milky_Way_center": {"M_msun": 1e12, "d_Mpc": 0.0083},
    "Coma": {"M_msun": 1.2e15, "d_Mpc": 99.0},
    "stacked_galaxies": {"M_msun": 1e13, "d_Mpc": 50.0},
}

SF_NEFF_NOTE = ("SF Delta N_eff is a heuristic (0.1 x the DW value), not a "
                "resonant-production calculation")


# ─────────────────────────────────────────────────────────────────
# Tool 0: describe
# ─────────────────────────────────────────────────────────────────

def describe_sterile_nu_tools() -> dict:
    """Describe this server's capabilities, the physics, and
    the observational data it uses.  Call this first."""
    return {
        "status": "ok",
        "files": [],
        "message": (
            "Sterile Neutrino -> BBN / X-ray Bridge Server\n"
            "=============================================\n\n"
            "Maps sterile-neutrino particle parameters (m_s, sin^2 2theta, "
            "production mechanism) to what the particle PRODUCES — Delta "
            "N_eff, relic density, radiative decay line, free streaming — "
            "and to the BBN and X-ray constraints that follow.\n\n"
            "TOOLS:\n"
            "  describe_sterile_nu_tools      — this help\n"
            "  predict_sterile_nu             — one (m_s, theta) point, all observables\n"
            "  predict_sterile_nu_grid        — many points in one call (table + CSV)\n"
            "  check_sterile_nu_sbl_anomaly   — the BEST anomaly vs BBN and Planck\n"
            "  predict_sterile_nu_xray_line   — X-ray line flux, one or all targets\n"
            "  scan_sterile_nu_parameters     — (m_s, theta) plane scan -> CSV/NPZ + BBN limit\n"
            "  plot_sterile_nu_exclusion      — exclusion plane from a scan file\n"
            "  plot_sterile_nu_bbn_impact     — Yp, D/H vs N_eff with data bands\n"
            "  list_sterile_nu_skills / load_sterile_nu_skill — workflow recipes\n\n"
            "BBN: Yp, D/H, Li7 from a fit to the PRyMordial network (<0.3%); "
            "exclusion uses Delta chi^2 vs the SM > 3.84 (95%, 1 dof). "
            "Valid for m_s below ~100 keV (relativistic at BBN). For MeV+ "
            "masses decay injection is NOT modeled — results there are "
            "flagged and are not constraints."
        ),
        "metadata": {
            "observations": {
                k: {"value": v.value, "sigma": v.sigma, "source": v.source}
                for k, v in OBS_DATA.items()
            },
            "bbn_fit_validity": BBN_FIT_RANGE,
            "bbn_exclusion_rule": f"delta_chi2_vs_sm > {DELTA_CHI2_95}",
            "xray_limits_count": len(XRAY_LIMITS),
            "xray_targets": list(XRAY_TARGETS),
            "sbl_best_experiment": SBL_BEST,
            "line_3p5_keV": LINE_35_KEV,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Tool 1: particle → cosmology prediction
# ─────────────────────────────────────────────────────────────────

def _production(m_s_eV, sin2_2theta, mechanism, lepton_asymmetry):
    if mechanism == "DW":
        return (dw_delta_neff(m_s_eV, sin2_2theta),
                dw_relic_density(m_s_eV, sin2_2theta))
    if mechanism == "SF":
        return (dw_delta_neff(m_s_eV, sin2_2theta) * 0.1,
                sf_relic_density(m_s_eV, sin2_2theta, lepton_asymmetry))
    raise ValueError(f"mechanism must be 'DW' or 'SF', got {mechanism!r}")


def _point(m_s_eV, sin2_2theta, mechanism="DW", lepton_asymmetry=1e-3,
           omega_b_h2=0.02237, tau_n=878.4) -> dict:
    """All observables for one point, as a flat dict (one table row)."""
    delta_neff, omega = _production(m_s_eV, sin2_2theta, mechanism,
                                     lepton_asymmetry)
    neff = 3.044 + delta_neff
    c2 = chi2_bbn(neff, omega_b_h2, tau_n)
    regime = mass_regime(m_s_eV)
    return {
        "m_s_eV": m_s_eV, "sin2_2theta": sin2_2theta, "mechanism": mechanism,
        "delta_neff": delta_neff, "neff": neff, "omega_s_h2": omega,
        "Yp": bbn_yp(neff, omega_b_h2, tau_n),
        "DH_x1e5": bbn_dh(neff, omega_b_h2, tau_n),
        "Li7H_x1e10": bbn_li7(neff, omega_b_h2, tau_n),
        "chi2_bbn": c2["chi2_total"],
        "delta_chi2_bbn": c2["delta_chi2_vs_sm"],
        "bbn_excluded_95": c2["excluded_95"],
        "E_gamma_keV": xray_line_energy_keV(m_s_eV),
        "decay_rate_per_s": radiative_decay_rate(m_s_eV, sin2_2theta),
        "lifetime_s": radiative_lifetime(m_s_eV, sin2_2theta),
        "free_streaming_kpc": free_streaming_length_kpc(m_s_eV, mechanism),
        "overcloses": bool(omega > 0.12),
        "regime": regime["regime"],
        "bbn_treatment_valid": regime["bbn_treatment_valid"],
        "_regime_note": regime["note"],
    }


def predict_from_particle_params(
    m_s_eV: float,
    sin2_2theta: float,
    mechanism: str = "DW",
    lepton_asymmetry: float = 1e-3,
    omega_b_h2: float = 0.02237,
    tau_n: float = 878.4,
) -> dict:
    """Given particle physics inputs, predict all cosmological observables.

    Parameters
    ----------
    m_s_eV : sterile neutrino mass in eV
    sin2_2theta : mixing parameter sin²(2θ)
    mechanism : "DW" (Dodelson-Widrow) or "SF" (Shi-Fuller)
    lepton_asymmetry : only used for SF mechanism
    omega_b_h2 : baryon density (default: Planck 2018)
    tau_n : neutron lifetime in seconds
    """
    r = _point(m_s_eV, sin2_2theta, mechanism, lepton_asymmetry,
               omega_b_h2, tau_n)
    omega, tau, lfs = r["omega_s_h2"], r["lifetime_s"], r["free_streaming_kpc"]
    is_all_dm = 0.10 < omega < 0.14
    is_subdominant_dm = 1e-4 < omega < 0.12

    msg_parts = []
    if not r["bbn_treatment_valid"]:
        msg_parts += [f"⚠ {r['regime']}: {r['_regime_note']}", ""]
    msg_parts += [
        f"Sterile neutrino: m_s = {m_s_eV:.3g} eV, sin²2θ = {sin2_2theta:.2g}",
        f"Mechanism: {mechanism}" + (f" ({SF_NEFF_NOTE})" if mechanism == "SF" else ""),
        "",
        "── What the sterile ν produces ──",
        f"ΔN_eff = {r['delta_neff']:.4g}  →  N_eff = {r['neff']:.4f}",
        f"Ω_s h² = {omega:.4g}  (obs DM: 0.120 ± 0.001)",
        "",
        "── BBN (PRyMordial-calibrated fit) ──",
        f"Y_p = {r['Yp']:.5f}  (obs: {OBS_DATA['Yp'].value} ± {OBS_DATA['Yp'].sigma})",
        f"D/H × 10⁵ = {r['DH_x1e5']:.4f}  (obs: {OBS_DATA['D_H'].value} ± {OBS_DATA['D_H'].sigma})",
        f"⁷Li/H × 10¹⁰ = {r['Li7H_x1e10']:.2f}  (obs: ~1.6 — the lithium problem)",
        f"Δχ²(BBN, vs SM) = {r['delta_chi2_bbn']:.2f}  "
        f"({'EXCLUDED' if r['bbn_excluded_95'] else 'allowed'} at 95%; "
        f"absolute χ² = {r['chi2_bbn']:.2f})",
        "",
        "── X-ray Decay ──",
        f"E_γ = {r['E_gamma_keV']:.4g} keV",
        f"Γ_γ = {r['decay_rate_per_s']:.3g} s⁻¹",
        f"τ_γ = {tau:.3g} s  ({tau / 3.156e7:.3g} yr)",
        "",
        "── Structure Formation ──",
        f"λ_fs = {lfs:.3g} kpc  ({'warm' if lfs > 10 else 'cold'} DM)",
    ]
    if r["overcloses"]:
        msg_parts.append(f"✗ Overproduces dark matter (Ω_s h² = {omega:.3g} > 0.12)")
    elif is_all_dm:
        msg_parts.append("★ This sterile ν could be ALL the dark matter")
    elif is_subdominant_dm:
        msg_parts.append(f"This sterile ν is subdominant DM ({omega / 0.12 * 100:.1f}% of Ω_DM)")

    return {
        "status": "ok",
        "files": [],
        "message": "\n".join(msg_parts),
        "metadata": {
            "input": {
                "m_s_eV": m_s_eV,
                "sin2_2theta": sin2_2theta,
                "mechanism": mechanism,
            },
            "regime": {"name": r["regime"],
                       "bbn_treatment_valid": r["bbn_treatment_valid"],
                       "note": r["_regime_note"]},
            "production": {
                "delta_neff": float(f"{r['delta_neff']:.6g}"),
                "neff_total": round(r["neff"], 6),
                "omega_s_h2": float(f"{omega:.6g}"),
                "overcloses": r["overcloses"],
            },
            "bbn": {
                "Yp": round(r["Yp"], 5),
                "DH_x1e5": round(r["DH_x1e5"], 4),
                "Li7H_x1e10": round(r["Li7H_x1e10"], 2),
                "chi2_total": round(r["chi2_bbn"], 4),
                "delta_chi2_vs_sm": round(r["delta_chi2_bbn"], 4),
                "excluded_95": r["bbn_excluded_95"],
                "consistent_2sigma": not r["bbn_excluded_95"],
                "method": "fit to PRyMordial (<0.3% within bbn_fit_validity)",
            },
            "xray": {
                "E_gamma_keV": round(r["E_gamma_keV"], 4),
                "decay_rate_per_s": float(f"{r['decay_rate_per_s']:.4g}"),
                "lifetime_s": float(f"{tau:.4g}"),
                "lifetime_yr": float(f"{tau / 3.156e7:.4g}"),
            },
            "structure": {
                "free_streaming_kpc": float(f"{lfs:.4g}"),
                "warm_or_cold": "warm" if lfs > 10 else "cold",
            },
            **({"caveat": SF_NEFF_NOTE} if mechanism == "SF" else {}),
        },
    }


# ─────────────────────────────────────────────────────────────────
# Tool 1b: many points in one call
# ─────────────────────────────────────────────────────────────────

_GRID_COLUMNS = ["m_s_eV", "sin2_2theta", "delta_neff", "neff", "omega_s_h2",
                 "Yp", "DH_x1e5", "Li7H_x1e10", "chi2_bbn", "delta_chi2_bbn",
                 "lifetime_s", "free_streaming_kpc"]


def predict_grid(
    masses_eV: list[float],
    sin2_2thetas: list[float],
    combine: str = "product",
    mechanism: str = "DW",
    lepton_asymmetry: float = 1e-3,
    output_dir: str = "output",
) -> dict:
    """Predict all observables for many (m_s, sin²2θ) points in one call.

    combine="product": every mass with every mixing; "zip": pair the lists
    element-wise (benchmark points). Writes a CSV and returns the rows
    inline (compact) plus regime/exclusion flags.
    """
    if combine == "zip":
        if len(masses_eV) != len(sin2_2thetas):
            raise ValueError("combine='zip' needs equal-length lists.")
        pts = list(zip(masses_eV, sin2_2thetas))
    elif combine == "product":
        pts = [(m, t) for m in masses_eV for t in sin2_2thetas]
    else:
        raise ValueError("combine must be 'product' or 'zip'.")
    if not pts or len(pts) > 400:
        raise ValueError(f"{len(pts)} points requested; allowed 1-400.")
    bad = [(m, t) for m, t in pts if not (m > 0 and 0 < t <= 1)]
    if bad:
        raise ValueError(f"Need m_s > 0 and 0 < sin2_2theta <= 1; got {bad[:3]}")

    rows = [_point(m, t, mechanism, lepton_asymmetry) for m, t in pts]
    out = _ensure_dir(_output_dir(output_dir))
    slug = _slug({"m": tuple(masses_eV), "t": tuple(sin2_2thetas),
                  "c": combine, "mech": mechanism, "L": lepton_asymmetry})
    csv_path = out / f"sterile_nu_grid_{mechanism}_{slug}.csv"
    flags = ("bbn_excluded_95", "overcloses", "bbn_treatment_valid")
    with open(csv_path, "w") as f:
        f.write(",".join(_GRID_COLUMNS + list(flags)) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.6g}" for c in _GRID_COLUMNS) + ","
                    + ",".join(str(int(r[c])) for c in flags) + "\n")

    compact = [{"m_s_eV": r["m_s_eV"], "sin2_2theta": r["sin2_2theta"],
                "delta_neff": float(f"{r['delta_neff']:.4g}"),
                "omega_s_h2": float(f"{r['omega_s_h2']:.4g}"),
                "delta_chi2_bbn": round(r["delta_chi2_bbn"], 3),
                "bbn_excluded_95": r["bbn_excluded_95"],
                "overcloses": r["overcloses"],
                "lifetime_s": float(f"{r['lifetime_s']:.4g}"),
                "regime": r["regime"]} for r in rows]
    invalid = sorted({r["regime"] for r in rows if not r["bbn_treatment_valid"]})
    msg = (f"Predicted {len(rows)} points [{mechanism}]: "
           f"{sum(r['bbn_excluded_95'] for r in rows)} BBN-excluded (95%), "
           f"{sum(r['overcloses'] for r in rows)} overclose. CSV: {csv_path}")
    if invalid:
        msg += (f"\n⚠ Points in {invalid}: BBN numbers there ignore decay "
                "injection and are NOT constraints.")
    return {
        "status": "ok",
        "files": [str(csv_path)],
        "message": msg,
        "metadata": {"rows": compact[:60], "n_rows": len(rows),
                     "truncated_inline": len(rows) > 60,
                     "columns": _GRID_COLUMNS + list(flags),
                     "regimes_outside_validity": invalid,
                     **({"caveat": SF_NEFF_NOTE} if mechanism == "SF" else {})},
    }


# ─────────────────────────────────────────────────────────────────
# Tool 2: BEST anomaly check
# ─────────────────────────────────────────────────────────────────

def check_sbl_anomaly() -> dict:
    """What does the BEST experiment's sterile ν imply for BBN?

    Uses BEST best-fit: Δm² = 1.3 eV², sin²2θ = 0.42
    → m_s ≈ 1.14 eV, fully thermalised → ΔN_eff ≈ 1
    """
    dm2 = SBL_BEST["delta_m2_eV2"]
    s22t = SBL_BEST["sin2_2theta"]
    m_s = np.sqrt(dm2)

    result = predict_from_particle_params(m_s, s22t, mechanism="DW")

    neff = result["metadata"]["production"]["neff_total"]
    planck = OBS_DATA["Neff_CMB"]
    tension_sigma = abs(neff - planck.value) / planck.sigma
    bbn = result["metadata"]["bbn"]

    extra = [
        "",
        "── BEST Anomaly Assessment ──",
        f"BEST best-fit: Δm² = {dm2} eV², sin²2θ = {s22t}",
        f"→ m_s ≈ {m_s:.2f} eV",
        "",
        f"This predicts N_eff = {neff:.3f}",
        f"Planck measures N_eff = {planck.value} ± {planck.sigma}",
        f"Tension with Planck: {tension_sigma:.1f}σ",
        f"BBN: Δχ² = {bbn['delta_chi2_vs_sm']:.1f} vs SM "
        f"({'excluded' if bbn['excluded_95'] else 'allowed'} at 95%)",
        "",
        f"CMB-S4 (σ ≈ 0.03) would see this at {abs(neff - 3.044) / 0.03:.0f}σ",
        "",
        f"Status: {SBL_BEST['status']}",
    ]
    result["message"] += "\n" + "\n".join(extra)
    result["metadata"]["sbl_assessment"] = {
        "tension_with_planck_sigma": round(tension_sigma, 2),
        "bbn_delta_chi2_vs_sm": bbn["delta_chi2_vs_sm"],
        "cmb_s4_detection_sigma": round(abs(neff - 3.044) / 0.03, 1),
        "status": SBL_BEST["status"],
    }
    return result


# ─────────────────────────────────────────────────────────────────
# Tool 3: X-ray signal prediction
# ─────────────────────────────────────────────────────────────────

def _xray_limits_violated(m_s_eV, sin2_2theta):
    m_keV = m_s_eV / 1e3
    return sorted({lim.source for lim in XRAY_LIMITS
                   if abs(m_keV - lim.m_s_keV) / lim.m_s_keV < 0.3
                   and sin2_2theta > lim.sin2_2theta_upper})


def predict_xray_signal(
    m_s_eV: float = 7.1e3,
    sin2_2theta: float = 7e-11,
    target: str = "Perseus",
) -> dict:
    """Predict X-ray decay line from sterile ν DM.

    Default: the 3.5 keV line candidate (m_s = 7.1 keV). target="all"
    returns every target in one call. Fluxes are whole-halo (no
    field-of-view or profile integration) and assume the sterile ν is all
    of the dark matter.
    """
    if target != "all" and target not in XRAY_TARGETS:
        raise ValueError(f"Unknown target {target!r}. Use one of "
                         f"{list(XRAY_TARGETS)} or 'all'.")
    names = list(XRAY_TARGETS) if target == "all" else [target]

    e_keV = xray_line_energy_keV(m_s_eV)
    rate = radiative_decay_rate(m_s_eV, sin2_2theta)
    tau = radiative_lifetime(m_s_eV, sin2_2theta)
    excluded_by = _xray_limits_violated(m_s_eV, sin2_2theta)
    fluxes = {n: xray_flux_from_halo(m_s_eV, sin2_2theta,
                                     XRAY_TARGETS[n]["M_msun"],
                                     XRAY_TARGETS[n]["d_Mpc"])
              for n in names}

    msg = [
        f"X-ray line prediction for {', '.join(names)}",
        f"m_s = {m_s_eV/1e3:.3g} keV, sin²2θ = {sin2_2theta:.2g}",
        "",
        f"E_γ = {e_keV:.4g} keV",
        f"Decay rate = {rate:.3g} s⁻¹",
        f"Lifetime = {tau:.3g} s ({tau/3.156e7:.3g} yr)",
        "",
    ]
    for n in sorted(names, key=lambda x: -fluxes[x]):
        t = XRAY_TARGETS[n]
        msg.append(f"{n}: M = {t['M_msun']:.2g} M☉, d = {t['d_Mpc']} Mpc "
                   f"→ flux = {fluxes[n]:.3g} ph/s/cm²")
    msg.append("(whole-halo flux, sterile ν = all DM; no field-of-view cut)")
    if excluded_by:
        msg.append(f"\n⚠ EXCLUDED by: {', '.join(excluded_by)}")
    else:
        msg.append("\n✓ Not excluded by tabulated X-ray limits (within 30% in mass)")
    if abs(m_s_eV - 7.1e3) < 500:
        msg.extend([
            "",
            "── 3.5 keV Line Context ──",
            f"Status: {LINE_35_KEV['status']}",
            f"Best-fit sin²2θ: {LINE_35_KEV['sin2_2theta_best']:.1g}",
            f"Range: {LINE_35_KEV['sin2_2theta_range']}",
        ])

    meta = {
        "E_gamma_keV": round(e_keV, 4),
        "lifetime_s": float(f"{tau:.4g}"),
        "flux_ph_s_cm2_by_target": {n: float(f"{v:.4g}") for n, v in fluxes.items()},
        "excluded_by": excluded_by,
        "available_targets": list(XRAY_TARGETS),
        "flux_model": "whole halo, f_DM = 1, no field-of-view cut",
    }
    if target != "all":
        meta["target"] = target
        meta["flux_ph_s_cm2"] = meta["flux_ph_s_cm2_by_target"][target]
    return {"status": "ok", "files": [], "message": "\n".join(msg),
            "metadata": meta}


# ─────────────────────────────────────────────────────────────────
# Tool 4: parameter space scan → CSV
# ─────────────────────────────────────────────────────────────────

_SCAN_DEFAULTS = {"m_min_eV": 1e2, "m_max_eV": 1e5, "theta_min": 1e-15,
                  "theta_max": 1e-5, "n_points": 60}


def _bbn_limit_by_mass(ms, s22t, dchi2, n_quote=8):
    """Largest sin²2θ still allowed by BBN at sampled masses."""
    idx = np.unique(np.linspace(0, len(ms) - 1, n_quote).astype(int))
    out = {}
    for i in idx:
        allowed = s22t[dchi2[i] <= DELTA_CHI2_95]
        if allowed.size == len(s22t):
            out[f"{ms[i]:.3g}"] = "no BBN limit in scanned range"
        else:
            out[f"{ms[i]:.3g}"] = float(f"{allowed.max():.3g}") if allowed.size else None
    return out


def scan_constraints(
    output_dir: str = "output",
    m_min_eV: float = 1e2,
    m_max_eV: float = 1e5,
    theta_min: float = 1e-15,
    theta_max: float = 1e-5,
    n_points: int = 60,
    mechanism: str = "DW",
) -> dict:
    """Scan the (m_s, sin²2θ) plane evaluating all constraints.

    Saves a CSV with columns:
      m_s_eV, sin2_2theta, neff, omega_h2, chi2_bbn, delta_chi2_bbn, lifetime_s
    The default ranges write scan_<mechanism>.{csv,npz}; any other range
    gets a parameter suffix so scans never overwrite each other.
    """
    if mechanism not in ("DW", "SF"):
        raise ValueError(f"mechanism must be 'DW' or 'SF', got {mechanism!r}")
    out = _ensure_dir(_output_dir(output_dir))
    ranges = {"m_min_eV": m_min_eV, "m_max_eV": m_max_eV,
              "theta_min": theta_min, "theta_max": theta_max,
              "n_points": n_points}

    data = scan_parameter_space(
        m_range_eV=(m_min_eV, m_max_eV),
        theta_range=(theta_min, theta_max),
        n_m=n_points,
        n_theta=n_points,
        mechanism=mechanism,
    )

    stem = f"scan_{mechanism}"
    if ranges != _SCAN_DEFAULTS:
        stem += f"_{_slug(ranges)}"
    csv_path = out / f"{stem}.csv"
    with open(csv_path, "w") as f:
        f.write("m_s_eV,sin2_2theta,neff,omega_h2,chi2_bbn,delta_chi2_bbn,lifetime_s\n")
        for i, ms in enumerate(data["m_s_eV"]):
            for j, s22t in enumerate(data["sin2_2theta"]):
                f.write(
                    f"{ms:.6e},{s22t:.6e},"
                    f"{data['neff'][i, j]:.6f},"
                    f"{data['omega_h2'][i, j]:.6e},"
                    f"{data['chi2_bbn'][i, j]:.6f},"
                    f"{data['delta_chi2_bbn'][i, j]:.6f},"
                    f"{data['lifetime_s'][i, j]:.6e}\n"
                )

    # Also save as npz for plotting tool
    npz_path = out / f"{stem}.npz"
    np.savez(npz_path, mechanism=mechanism, **data)

    ms, s22t, dchi2 = data["m_s_eV"], data["sin2_2theta"], data["delta_chi2_bbn"]
    n_excluded = int(np.sum(dchi2 > DELTA_CHI2_95))
    n_over = int(np.sum(data["omega_h2"] > 0.12))
    n_total = n_points * n_points
    invalid = sorted({mass_regime(m)["regime"] for m in ms
                      if not mass_regime(m)["bbn_treatment_valid"]})
    msg = (f"Scanned {n_total} points in (m_s, sin²2θ) plane [{mechanism}]\n"
           f"BBN-excluded (Δχ² > {DELTA_CHI2_95}, 95%): {n_excluded}/{n_total}; "
           f"overclosing (Ω_s h² > 0.12): {n_over}/{n_total}\n"
           f"Saved to {csv_path} and {npz_path}")
    if invalid:
        msg += (f"\n⚠ Range includes {invalid}: outside the validity of the "
                "Delta-N_eff-only BBN treatment (MeV+: decay injection not "
                "modeled) — not a constraint there.")
    return {
        "status": "ok",
        "files": [str(csv_path), str(npz_path)],
        "message": msg,
        "metadata": {
            "n_points": n_total,
            "n_bbn_excluded": n_excluded,
            "n_overclosing": n_over,
            "bbn_max_allowed_sin2_2theta_by_mass_eV": _bbn_limit_by_mass(ms, s22t, dchi2),
            "exclusion_rule": f"delta_chi2_bbn > {DELTA_CHI2_95}",
            "regimes_outside_validity": invalid,
            "mechanism": mechanism,
            **({"caveat": SF_NEFF_NOTE} if mechanism == "SF" else {}),
        },
    }


# ─────────────────────────────────────────────────────────────────
# Tool 5: exclusion plot
# ─────────────────────────────────────────────────────────────────

def plot_exclusion(
    scan_file: str | None = None,
    output_dir: str = "output",
    title: str | None = None,
    save_pdf: bool = False,
) -> dict:
    """Plot the exclusion plane from a scan file.

    Overlays: BBN exclusion (Δχ² vs SM), overclosure (Ω_s h² > 0.12) and the
    Ω_DM line, X-ray limits, the Lyman-α thermal-WDM bound, the 3.5 keV
    line, and hatching where the BBN treatment is outside its validity.
    scan_file=None uses the newest scan_*.npz in output_dir.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from tools.plotting import (BAND_ALPHA, MUTED, PALETTE, legend_below,
                                rc_params, save, wrap)

    out = _ensure_dir(_output_dir(output_dir))
    if scan_file is None:
        scans = sorted(out.glob("scan_*.npz"), key=lambda p: p.stat().st_mtime)
        if not scans:
            raise FileNotFoundError(f"No scan_*.npz in {out}; run a scan first.")
        scan_file = str(scans[-1])
    scan_path = Path(scan_file).expanduser()
    # A scan writes a CSV (listed first) and an .npz with the full grid;
    # accept either, since agents naturally pass the CSV.
    if scan_path.suffix != ".npz":
        sibling = scan_path.with_suffix(".npz")
        if not sibling.exists():
            raise ValueError(
                f"{scan_path.name} is not a scan grid: pass the scan_*.npz "
                "(or scan_*.csv next to it) written by scan_sterile_nu_parameters.")
        scan_path = sibling
    data = np.load(scan_path)
    ms = data["m_s_eV"]
    s22t = data["sin2_2theta"]
    omega = data["omega_h2"]
    if "delta_chi2_bbn" in data:
        dchi2 = data["delta_chi2_bbn"]
    else:  # scans written before the Δχ² column existed: recompute
        dchi2 = np.vectorize(lambda n: chi2_bbn(n)["delta_chi2_vs_sm"])(data["neff"])
    mechanism = str(data["mechanism"]) if "mechanism" in data else (
        "SF" if "SF" in scan_path.stem else "DW")

    M, T = np.meshgrid(ms / 1e3, s22t, indexing="ij")
    handles, labels = [], []
    c_bbn, c_dm, c_x, c_ly, c_line = PALETTE[1], PALETTE[0], "#333333", PALETTE[2], PALETTE[3]

    with plt.rc_context(rc_params()):
        fig, ax = plt.subplots(figsize=(6.8, 5.4), layout="constrained")

        if np.any(dchi2 > DELTA_CHI2_95):
            ax.contourf(M, T, dchi2, levels=[DELTA_CHI2_95, 1e9],
                        colors=[c_bbn], alpha=BAND_ALPHA)
            ax.contour(M, T, dchi2, levels=[DELTA_CHI2_95], colors=[c_bbn],
                       linewidths=1.8)
            handles.append(Patch(facecolor=c_bbn, alpha=0.4, edgecolor=c_bbn))
            labels.append(r"BBN excluded ($\Delta\chi^2>3.84$)")

        if np.any(omega > 0.12):
            ax.contourf(M, T, omega, levels=[0.12, 1e30], colors=[c_dm],
                        alpha=0.10)
            ax.contour(M, T, omega, levels=[0.12], colors=[c_dm],
                       linewidths=1.8, linestyles=["--"])
            handles.append(Line2D([], [], color=c_dm, ls="--"))
            labels.append(r"$\Omega_s h^2 = 0.12$ (shaded: overcloses)")

        curves = {}
        for src, ls in (("NuSTAR", "-"), ("XMM", "-.")):
            pts = sorted((l.m_s_keV, l.sin2_2theta_upper)
                         for l in XRAY_LIMITS if src in l.source)
            if pts:
                xm, xt = map(np.array, zip(*pts))
                curves[src] = (xm, xt)
                ax.plot(xm, xt, color=c_x, ls=ls, lw=1.6)
                handles.append(Line2D([], [], color=c_x, ls=ls))
                labels.append({"NuSTAR": "NuSTAR M31 (Roach+ 2020)",
                               "XMM": "XMM blank sky (Dessert+ 2020)"}[src])
        if curves:
            # one excluded region: above the strongest limit at each mass
            lo = min(c[0][0] for c in curves.values())
            hi = max(c[0][-1] for c in curves.values())
            grid = np.geomspace(lo, hi, 200)
            env = np.min([np.where((grid >= xm[0]) & (grid <= xm[-1]),
                                   np.exp(np.interp(np.log(grid), np.log(xm), np.log(xt))),
                                   np.inf) for xm, xt in curves.values()], axis=0)
            ax.fill_between(grid, env, 1e2, color=c_x, alpha=0.10, lw=0)
            handles.append(Patch(facecolor=c_x, alpha=0.25))
            labels.append("X-ray excluded")

        ax.axvline(LYMAN_ALPHA_LOWER_MASS_KEV, color=c_ly, ls=":", lw=1.8)
        handles.append(Line2D([], [], color=c_ly, ls=":"))
        labels.append(rf"Ly-$\alpha$: thermal WDM $m>{LYMAN_ALPHA_LOWER_MASS_KEV}$ keV "
                      "(DW-equivalent bound is higher)")

        ax.plot(LINE_35_KEV["m_s_eV"] / 1e3, LINE_35_KEV["sin2_2theta_best"],
                marker="*", ms=14, color=c_line, ls="none",
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        handles.append(Line2D([], [], marker="*", ms=12, color=c_line, ls="none"))
        labels.append("3.5 keV line (debated)")

        invalid = ms >= DECAY_PHYSICS_MIN_EV
        if invalid.any():
            ax.axvspan(ms[invalid][0] / 1e3, ms[-1] / 1e3, facecolor="none",
                       edgecolor=MUTED, hatch="//", lw=0)
            handles.append(Patch(facecolor="none", edgecolor=MUTED, hatch="//"))
            labels.append("BBN treatment invalid (decays not modeled)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(ms[0] / 1e3, ms[-1] / 1e3)
        ax.set_ylim(s22t[0], s22t[-1])
        ax.set_xlabel(r"$m_s$ [keV]")
        ax.set_ylabel(r"$\sin^2 2\theta$")
        ax.set_title(wrap(title or f"Sterile-neutrino constraints ({mechanism} production)", 62),
                     loc="left", pad=8)
        legend_below(fig, handles, labels, ncol=2)

        fig_path = out / f"exclusion_{scan_path.stem}.png"
        files = save(fig, fig_path, save_pdf)
        plt.close(fig)

    return {
        "status": "ok",
        "files": files,
        "message": f"Exclusion plot saved to {fig_path} (from {scan_path.name}).",
        "metadata": {"scan_file": str(scan_path), "mechanism": mechanism,
                     "bbn_rule": f"delta_chi2 > {DELTA_CHI2_95}",
                     "hatched_invalid_region": bool(invalid.any())},
    }


# ─────────────────────────────────────────────────────────────────
# Tool 6: BBN vs N_eff plot with data
# ─────────────────────────────────────────────────────────────────

def plot_bbn_vs_neff(
    output_dir: str = "output",
    neff_min: float = 2.0,
    neff_max: float = 5.0,
    n_points: int = 200,
    save_pdf: bool = False,
) -> dict:
    """Plot Y_p and D/H as functions of N_eff, overlaid with
    observational bands from primordial abundance measurements
    and Planck N_eff, marking the 95% BBN limit on ΔN_eff.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy.optimize import brentq

    from tools.plotting import (BAND_ALPHA, INK, MUTED, PALETTE,
                                legend_below, rc_params, save)

    out = _ensure_dir(_output_dir(output_dir))

    neff_arr = np.linspace(neff_min, neff_max, n_points)
    yp_arr = np.array([bbn_yp(n) for n in neff_arr])
    dh_arr = np.array([bbn_dh(n) for n in neff_arr])
    yp_obs, dh_obs, neff_obs = OBS_DATA["Yp"], OBS_DATA["D_H"], OBS_DATA["Neff_CMB"]
    dn_limit = brentq(lambda d: chi2_bbn(3.044 + d)["delta_chi2_vs_sm"] - DELTA_CHI2_95,
                      1e-4, 3.0)

    c_pred, c_obs, c_planck, c_lim = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]
    with plt.rc_context(rc_params()):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 6.6), sharex=True,
                                       layout="constrained")
        for ax, y, obs in ((ax1, yp_arr, yp_obs), (ax2, dh_arr, dh_obs)):
            ax.axvspan(neff_obs.value - neff_obs.sigma,
                       neff_obs.value + neff_obs.sigma,
                       color=c_planck, alpha=0.12, lw=0)
            ax.axhspan(obs.value - obs.sigma, obs.value + obs.sigma,
                       color=c_obs, alpha=BAND_ALPHA, lw=0)
            ax.plot(neff_arr, y, color=c_pred, lw=2.0, zorder=3)
            ax.axvline(3.044, color=MUTED, ls=":", lw=1.2)
            ax.axvline(4.044, color=INK, ls="--", lw=1.2)
            if neff_min < 3.044 + dn_limit < neff_max:
                ax.axvline(3.044 + dn_limit, color=c_lim, ls="-.", lw=1.4)
        ax1.set_ylabel(r"$Y_p$")
        ax2.set_ylabel(r"D/H $\times 10^5$")
        ax2.set_xlabel(r"$N_{\rm eff}$")
        ax2.set_xlim(neff_min, neff_max)
        ax1.set_title(r"BBN response to extra radiation $\Delta N_{\rm eff}$",
                      loc="left", pad=8)

        handles = [Line2D([], [], color=c_pred, lw=2),
                   Patch(facecolor=c_obs, alpha=0.4),
                   Patch(facecolor=c_planck, alpha=0.3),
                   Line2D([], [], color=MUTED, ls=":"),
                   Line2D([], [], color=INK, ls="--"),
                   Line2D([], [], color=c_lim, ls="-.")]
        labels = ["BBN (PRyMordial-calibrated fit)",
                  r"Observed (1$\sigma$): $Y_p$ Aver+21, D/H Cooke+18",
                  rf"Planck $N_{{\rm eff}} = {neff_obs.value} \pm {neff_obs.sigma}$",
                  r"SM: $N_{\rm eff} = 3.044$",
                  r"+1 thermalised sterile $\nu$",
                  rf"BBN 95% limit: $\Delta N_{{\rm eff}} < {dn_limit:.2f}$"]
        legend_below(fig, handles, labels, ncol=2)

        suffix = ("" if (neff_min, neff_max) == (2.0, 5.0)
                  else f"_{_slug({'a': neff_min, 'b': neff_max})}")
        fig_path = out / f"bbn_vs_neff{suffix}.png"
        files = save(fig, fig_path, save_pdf)
        plt.close(fig)

    yp1, dh1 = bbn_yp(4.044), bbn_dh(4.044)
    return {
        "status": "ok",
        "files": files,
        "message": (
            f"Plot saved to {fig_path}\n\n"
            f"Key result: a fully thermalised sterile ν (N_eff = 4.044) "
            f"predicts Y_p = {yp1:.4f} and D/H×10⁵ = {dh1:.3f}: "
            f"{abs(yp1 - yp_obs.value) / yp_obs.sigma:.1f}σ above observed Y_p. "
            f"BBN alone limits ΔN_eff < {dn_limit:.2f} (95%, Δχ² vs SM)."
        ),
        "metadata": {
            "sm_predictions": {
                "Yp_at_3044": round(bbn_yp(3.044), 5),
                "DH5_at_3044": round(bbn_dh(3.044), 4),
            },
            "one_sterile_predictions": {
                "Yp_at_4044": round(yp1, 5),
                "DH5_at_4044": round(dh1, 4),
            },
            "bbn_delta_neff_limit_95": round(dn_limit, 3),
            "slopes_near_sm": {
                "dYp_dNeff": round((bbn_yp(3.144) - bbn_yp(2.944)) / 0.2, 5),
                "dDH5_dNeff": round((bbn_dh(3.144) - bbn_dh(2.944)) / 0.2, 4)},
        },
    }
