"""
MCP tool functions for the sterile-neutrino-BBN bridge server.

Every tool returns an ArtifactResult dict:
    {status, files, message, metadata}
Arrays pass between tools as file paths, never through the context window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from tools.physics import (
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
    radiative_decay_rate,
    radiative_lifetime,
    scan_parameter_space,
    sf_relic_density,
    xray_flux_from_halo,
    xray_line_energy_keV,
)

DEFAULT_OUTPUT = Path("output")


def _ensure_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


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
            "Sterile Neutrino ↔ BBN Bridge Server\n"
            "=====================================\n\n"
            "This server bridges particle physics (sterile neutrino "
            "parameters) with cosmological observables (BBN abundances, "
            "CMB N_eff, X-ray lines).\n\n"
            "TOOLS:\n"
            "  describe_sterile_nu_tools()  — this help\n"
            "  predict_from_particle_params() — particle → cosmo predictions\n"
            "  check_sbl_anomaly()           — what BEST's sterile ν means for BBN\n"
            "  predict_xray_signal()         — X-ray line from keV sterile ν DM\n"
            "  scan_constraints()            — scan (m_s, θ) plane, save CSV\n"
            "  plot_exclusion()              — exclusion plot from scan data\n"
            "  plot_bbn_vs_neff()            — Yp, D/H vs N_eff with data bands\n\n"
            "OBSERVATIONAL DATA USED:\n"
        ),
        "metadata": {
            "observations": {
                k: {"value": v.value, "sigma": v.sigma, "source": v.source}
                for k, v in OBS_DATA.items()
            },
            "xray_limits_count": len(XRAY_LIMITS),
            "sbl_best_experiment": SBL_BEST,
            "line_3p5_keV": LINE_35_KEV,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Tool 1: particle → cosmology prediction
# ─────────────────────────────────────────────────────────────────

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
    # Production
    if mechanism == "DW":
        delta_neff = dw_delta_neff(m_s_eV, sin2_2theta)
        omega = dw_relic_density(m_s_eV, sin2_2theta)
    else:
        delta_neff = dw_delta_neff(m_s_eV, sin2_2theta) * 0.1
        omega = sf_relic_density(m_s_eV, sin2_2theta, lepton_asymmetry)

    neff = 3.044 + delta_neff

    # BBN
    yp = bbn_yp(neff, omega_b_h2, tau_n)
    dh = bbn_dh(neff, omega_b_h2, tau_n)
    li7 = bbn_li7(neff, omega_b_h2)
    bbn_chi2 = chi2_bbn(neff, omega_b_h2)

    # X-ray
    e_xray = xray_line_energy_keV(m_s_eV)
    gamma = radiative_decay_rate(m_s_eV, sin2_2theta)
    tau = radiative_lifetime(m_s_eV, sin2_2theta)

    # Structure
    lfs = free_streaming_length_kpc(m_s_eV, mechanism)

    # Is it viable DM?
    is_all_dm = 0.10 < omega < 0.14
    is_subdominant_dm = omega < 0.12 and omega > 1e-4

    msg_parts = [
        f"Sterile neutrino: m_s = {m_s_eV:.2g} eV, sin²2θ = {sin2_2theta:.2g}",
        f"Mechanism: {mechanism}",
        f"",
        f"── Particle → Cosmology Bridge ──",
        f"ΔN_eff = {delta_neff:.4f}  →  N_eff = {neff:.4f}",
        f"Ω_s h² = {omega:.4g}  (obs DM: 0.120 ± 0.001)",
        f"",
        f"── BBN Predictions ──",
        f"Y_p = {yp:.5f}  (obs: {OBS_DATA['Yp'].value} ± {OBS_DATA['Yp'].sigma})",
        f"D/H × 10⁵ = {dh:.4f}  (obs: {OBS_DATA['D_H'].value} ± {OBS_DATA['D_H'].sigma})",
        f"⁷Li/H × 10¹⁰ = {li7:.2f}  (obs: ~1.6 — the lithium problem)",
        f"BBN χ² = {bbn_chi2['chi2_total']:.2f}  ({'CONSISTENT' if bbn_chi2['consistent_2sigma'] else 'EXCLUDED'} at 2σ)",
        f"",
        f"── X-ray Decay ──",
        f"E_γ = {e_xray:.3f} keV",
        f"Γ_γ = {gamma:.3g} s⁻¹",
        f"τ_γ = {tau:.3g} s  ({tau / 3.156e7:.3g} yr)",
        f"",
        f"── Structure Formation ──",
        f"λ_fs = {lfs:.1f} kpc  ({'warm' if lfs > 10 else 'cold'} DM)",
    ]

    if is_all_dm:
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
            "production": {
                "delta_neff": round(delta_neff, 6),
                "neff_total": round(neff, 6),
                "omega_s_h2": float(f"{omega:.6g}"),
            },
            "bbn": {
                "Yp": round(yp, 5),
                "DH_x1e5": round(dh, 4),
                "Li7H_x1e10": round(li7, 2),
                "chi2_total": round(bbn_chi2["chi2_total"], 4),
                "consistent_2sigma": bbn_chi2["consistent_2sigma"],
            },
            "xray": {
                "E_gamma_keV": round(e_xray, 4),
                "decay_rate_per_s": float(f"{gamma:.4g}"),
                "lifetime_s": float(f"{tau:.4g}"),
                "lifetime_yr": float(f"{tau / 3.156e7:.4g}"),
            },
            "structure": {
                "free_streaming_kpc": round(lfs, 2),
                "warm_or_cold": "warm" if lfs > 10 else "cold",
            },
        },
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

    extra = [
        "",
        "── BEST Anomaly Assessment ──",
        f"BEST best-fit: Δm² = {dm2} eV², sin²2θ = {s22t}",
        f"→ m_s ≈ {m_s:.2f} eV",
        f"",
        f"This predicts N_eff = {neff:.3f}",
        f"Planck measures N_eff = {planck.value} ± {planck.sigma}",
        f"Tension: {tension_sigma:.1f}σ",
        f"",
        f"CMB-S4 (σ ≈ 0.03) would see this at {abs(neff - 3.044) / 0.03:.0f}σ",
        f"",
        f"Status: {SBL_BEST['status']}",
    ]
    result["message"] += "\n" + "\n".join(extra)
    result["metadata"]["sbl_assessment"] = {
        "tension_with_planck_sigma": round(tension_sigma, 2),
        "cmb_s4_detection_sigma": round(abs(neff - 3.044) / 0.03, 1),
        "status": SBL_BEST["status"],
    }
    return result


# ─────────────────────────────────────────────────────────────────
# Tool 3: X-ray signal prediction
# ─────────────────────────────────────────────────────────────────

def predict_xray_signal(
    m_s_eV: float = 7.1e3,
    sin2_2theta: float = 7e-11,
    target: str = "Perseus",
) -> dict:
    """Predict X-ray decay line from sterile ν DM.

    Default: the 3.5 keV line candidate (m_s = 7.1 keV).
    """
    targets = {
        "Perseus": {"M_msun": 6.65e14, "d_Mpc": 77.7},
        "M31": {"M_msun": 1.3e12, "d_Mpc": 0.78},
        "Milky_Way_center": {"M_msun": 1e12, "d_Mpc": 0.0083},
        "Coma": {"M_msun": 1.2e15, "d_Mpc": 99.0},
        "stacked_galaxies": {"M_msun": 1e13, "d_Mpc": 50.0},
    }

    if target not in targets:
        target = "Perseus"

    t = targets[target]
    flux = xray_flux_from_halo(
        m_s_eV, sin2_2theta, t["M_msun"], t["d_Mpc"]
    )
    e_keV = xray_line_energy_keV(m_s_eV)
    rate = radiative_decay_rate(m_s_eV, sin2_2theta)
    tau = radiative_lifetime(m_s_eV, sin2_2theta)

    # Check against X-ray limits
    m_keV = m_s_eV / 1e3
    excluded_by = []
    for lim in XRAY_LIMITS:
        if abs(m_keV - lim.m_s_keV) / lim.m_s_keV < 0.3:
            if sin2_2theta > lim.sin2_2theta_upper:
                excluded_by.append(lim.source)

    msg = [
        f"X-ray line prediction for {target}",
        f"m_s = {m_s_eV/1e3:.2f} keV, sin²2θ = {sin2_2theta:.2g}",
        f"",
        f"E_γ = {e_keV:.3f} keV",
        f"Decay rate = {rate:.3g} s⁻¹",
        f"Lifetime = {tau:.3g} s ({tau/3.156e7:.3g} yr)",
        f"",
        f"Target: {target} (M = {t['M_msun']:.2g} M☉, d = {t['d_Mpc']} Mpc)",
        f"Predicted flux = {flux:.3g} photons/s/cm²",
    ]

    if excluded_by:
        msg.append(f"\n⚠ EXCLUDED by: {', '.join(set(excluded_by))}")
    else:
        msg.append("\n✓ Not excluded by current X-ray limits")

    if abs(m_s_eV - 7.1e3) < 500:
        msg.extend([
            "",
            "── 3.5 keV Line Context ──",
            f"Status: {LINE_35_KEV['status']}",
            f"Best-fit sin²2θ: {LINE_35_KEV['sin2_2theta_best']:.1g}",
            f"Range: {LINE_35_KEV['sin2_2theta_range']}",
        ])

    return {
        "status": "ok",
        "files": [],
        "message": "\n".join(msg),
        "metadata": {
            "E_gamma_keV": round(e_keV, 4),
            "flux_ph_s_cm2": float(f"{flux:.4g}"),
            "target": target,
            "excluded_by": list(set(excluded_by)),
            "available_targets": list(targets.keys()),
        },
    }


# ─────────────────────────────────────────────────────────────────
# Tool 4: parameter space scan → CSV
# ─────────────────────────────────────────────────────────────────

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
      m_s_eV, sin2_2theta, neff, omega_h2, chi2_bbn, lifetime_s
    """
    out = _ensure_dir(Path(output_dir))

    data = scan_parameter_space(
        m_range_eV=(m_min_eV, m_max_eV),
        theta_range=(theta_min, theta_max),
        n_m=n_points,
        n_theta=n_points,
        mechanism=mechanism,
    )

    csv_path = out / f"scan_{mechanism}.csv"
    with open(csv_path, "w") as f:
        f.write("m_s_eV,sin2_2theta,neff,omega_h2,chi2_bbn,lifetime_s\n")
        for i, ms in enumerate(data["m_s_eV"]):
            for j, s22t in enumerate(data["sin2_2theta"]):
                f.write(
                    f"{ms:.6e},{s22t:.6e},"
                    f"{data['neff'][i, j]:.6f},"
                    f"{data['omega_h2'][i, j]:.6e},"
                    f"{data['chi2_bbn'][i, j]:.6f},"
                    f"{data['lifetime_s'][i, j]:.6e}\n"
                )

    # Also save as npz for plotting tool
    npz_path = out / f"scan_{mechanism}.npz"
    np.savez(
        npz_path,
        m_s_eV=data["m_s_eV"],
        sin2_2theta=data["sin2_2theta"],
        neff=data["neff"],
        omega_h2=data["omega_h2"],
        chi2_bbn=data["chi2_bbn"],
        lifetime_s=data["lifetime_s"],
    )

    n_excluded = int(np.sum(data["chi2_bbn"] > 6.18))
    n_total = n_points * n_points

    return {
        "status": "ok",
        "files": [str(csv_path), str(npz_path)],
        "message": (
            f"Scanned {n_total} points in (m_s, sin²2θ) plane [{mechanism}]\n"
            f"BBN-excluded (2σ): {n_excluded}/{n_total} points\n"
            f"Saved to {csv_path} and {npz_path}"
        ),
        "metadata": {
            "n_points": n_total,
            "n_bbn_excluded": n_excluded,
            "mechanism": mechanism,
        },
    }


# ─────────────────────────────────────────────────────────────────
# Tool 5: exclusion plot
# ─────────────────────────────────────────────────────────────────

def plot_exclusion(
    scan_file: str = "output/scan_DW.npz",
    output_dir: str = "output",
) -> dict:
    """Plot the exclusion contour from a scan.

    Overlays: BBN exclusion, X-ray limits, Lyman-α lower bound,
    relic density = Ω_DM contour, and the 3.5 keV line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_dir(Path(output_dir))
    data = np.load(scan_file)
    ms = data["m_s_eV"]
    s22t = data["sin2_2theta"]
    chi2 = data["chi2_bbn"]
    omega = data["omega_h2"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    M, T = np.meshgrid(ms / 1e3, s22t, indexing="ij")

    # BBN exclusion (chi2 > 6.18 at 2 dof 95% CL)
    ax.contourf(
        M, T, chi2, levels=[6.18, 1e6],
        colors=["#ff9999"], alpha=0.4,
    )
    ax.contour(
        M, T, chi2, levels=[6.18],
        colors=["red"], linewidths=2,
    )

    # Relic density contour
    ax.contour(
        M, T, omega, levels=[0.12],
        colors=["blue"], linewidths=2, linestyles=["--"],
    )

    # X-ray limits
    xray_m = [l.m_s_keV for l in XRAY_LIMITS if "NuSTAR" in l.source]
    xray_t = [l.sin2_2theta_upper for l in XRAY_LIMITS if "NuSTAR" in l.source]
    if xray_m:
        idx = np.argsort(xray_m)
        ax.plot(
            [xray_m[i] for i in idx],
            [xray_t[i] for i in idx],
            "k-", lw=2, label="NuSTAR (Roach+ 2020)",
        )
        ax.fill_between(
            [xray_m[i] for i in idx],
            [xray_t[i] for i in idx],
            1e-2, alpha=0.15, color="gray",
        )

    # Lyman-α lower bound
    ax.axvline(
        LYMAN_ALPHA_LOWER_MASS_KEV,
        color="green", ls=":", lw=2, label=f"Ly-α lower bound ({LYMAN_ALPHA_LOWER_MASS_KEV} keV)",
    )

    # 3.5 keV line
    ax.plot(
        LINE_35_KEV["m_s_eV"] / 1e3,
        LINE_35_KEV["sin2_2theta_best"],
        "r*", ms=15, label="3.5 keV line (debated)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_s$ [keV]", fontsize=14)
    ax.set_ylabel(r"$\sin^2(2\theta)$", fontsize=14)
    ax.set_title(
        "Sterile Neutrino Constraints: BBN + X-ray + Structure",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="upper right")

    # Annotation
    ax.text(
        0.03, 0.03,
        "Red shaded: BBN-excluded (2σ)\n"
        "Blue dashed: Ω_s h² = 0.12\n"
        "Gray: X-ray excluded",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig_path = out / "exclusion_plot.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "ok",
        "files": [str(fig_path)],
        "message": f"Exclusion plot saved to {fig_path}",
        "metadata": {},
    }


# ─────────────────────────────────────────────────────────────────
# Tool 6: BBN vs N_eff plot with data
# ─────────────────────────────────────────────────────────────────

def plot_bbn_vs_neff(
    output_dir: str = "output",
    neff_min: float = 2.0,
    neff_max: float = 5.0,
    n_points: int = 200,
) -> dict:
    """Plot Y_p and D/H as functions of N_eff, overlaid with
    observational bands from primordial abundance measurements
    and Planck N_eff.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _ensure_dir(Path(output_dir))

    neff_arr = np.linspace(neff_min, neff_max, n_points)
    yp_arr = np.array([bbn_yp(n) for n in neff_arr])
    dh_arr = np.array([bbn_dh(n) for n in neff_arr])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Y_p panel
    ax1.plot(neff_arr, yp_arr, "b-", lw=2, label="BBN prediction")
    yp_obs = OBS_DATA["Yp"]
    ax1.axhspan(
        yp_obs.value - yp_obs.sigma,
        yp_obs.value + yp_obs.sigma,
        alpha=0.3, color="orange",
        label=f"Observed: {yp_obs.value} ± {yp_obs.sigma}",
    )
    ax1.axhline(yp_obs.value, color="orange", ls="--", alpha=0.7)

    # Planck N_eff band
    neff_obs = OBS_DATA["Neff_CMB"]
    ax1.axvspan(
        neff_obs.value - neff_obs.sigma,
        neff_obs.value + neff_obs.sigma,
        alpha=0.15, color="green",
        label=f"Planck N_eff: {neff_obs.value} ± {neff_obs.sigma}",
    )

    ax1.axvline(3.044, color="gray", ls=":", alpha=0.5, label="SM: N_eff = 3.044")
    ax1.axvline(4.044, color="red", ls=":", alpha=0.5, label="+1 sterile ν")
    ax1.set_ylabel(r"$Y_p$ (He-4 mass fraction)", fontsize=13)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.set_title("BBN Abundances vs N_eff — Sterile Neutrino Impact", fontsize=14)

    # D/H panel
    ax2.plot(neff_arr, dh_arr, "r-", lw=2, label="BBN prediction")
    dh_obs = OBS_DATA["D_H"]
    ax2.axhspan(
        dh_obs.value - dh_obs.sigma,
        dh_obs.value + dh_obs.sigma,
        alpha=0.3, color="cyan",
        label=f"Observed: {dh_obs.value} ± {dh_obs.sigma}",
    )
    ax2.axhline(dh_obs.value, color="cyan", ls="--", alpha=0.7)
    ax2.axvspan(
        neff_obs.value - neff_obs.sigma,
        neff_obs.value + neff_obs.sigma,
        alpha=0.15, color="green",
    )
    ax2.axvline(3.044, color="gray", ls=":", alpha=0.5)
    ax2.axvline(4.044, color="red", ls=":", alpha=0.5)

    ax2.set_xlabel(r"$N_{\rm eff}$", fontsize=13)
    ax2.set_ylabel(r"D/H $\times 10^5$", fontsize=13)
    ax2.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    fig_path = out / "bbn_vs_neff.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "status": "ok",
        "files": [str(fig_path)],
        "message": (
            f"Plot saved to {fig_path}\n\n"
            f"Key result: A fully thermalised sterile ν (N_eff = 4.044) "
            f"predicts Y_p = {bbn_yp(4.044):.4f} and D/H×10⁵ = {bbn_dh(4.044):.3f}.\n"
            f"This is in {abs(bbn_yp(4.044) - yp_obs.value) / yp_obs.sigma:.1f}σ "
            f"tension with observed Y_p."
        ),
        "metadata": {
            "sm_predictions": {
                "Yp_at_3044": round(bbn_yp(3.044), 5),
                "DH5_at_3044": round(bbn_dh(3.044), 4),
            },
            "one_sterile_predictions": {
                "Yp_at_4044": round(bbn_yp(4.044), 5),
                "DH5_at_4044": round(bbn_dh(4.044), 4),
            },
        },
    }
