"""MCP server entry point for sterile-nu-bbn-server (MCP SDK v2)."""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from tools.sterile_tools import (
    describe_sterile_nu_tools,
    predict_from_particle_params,
    check_sbl_anomaly,
    predict_xray_signal,
    scan_constraints,
    plot_exclusion,
    plot_bbn_vs_neff,
)

from mcp_server.dispatch import (
    set_dispatch,
    get_dispatch,
    auth_status,
    export_dispatch_pack,
)

mcp = FastMCP(
    name="sterile-nu-bbn-server",
    instructions=(
        "Bridges sterile neutrino particle physics with BBN cosmology. "
        "Predicts N_eff, primordial abundances, X-ray lines, and "
        "compares against real observational data."
    ),
)


@mcp.tool()
def describe_tools() -> str:
    """Describe this server's capabilities, the physics, and the observational data it uses. Call this first."""
    import json
    result = describe_sterile_nu_tools()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def predict_sterile_nu(
    m_s_eV: float,
    sin2_2theta: float,
    mechanism: str = "DW",
    lepton_asymmetry: float = 1e-3,
    omega_b_h2: float = 0.02237,
    tau_n: float = 878.4,
) -> str:
    """Given particle physics inputs (mass, mixing angle), predict all cosmological observables.

    Parameters:
        m_s_eV: sterile neutrino mass in eV (e.g. 1.14 for BEST, 7100 for 3.5 keV line)
        sin2_2theta: mixing parameter sin²(2θ)
        mechanism: "DW" (Dodelson-Widrow) or "SF" (Shi-Fuller resonant)
        lepton_asymmetry: lepton number asymmetry (only for SF mechanism)
        omega_b_h2: baryon density parameter (default: Planck 2018)
        tau_n: neutron lifetime in seconds (default: 878.4)
    """
    import json
    result = predict_from_particle_params(
        m_s_eV, sin2_2theta, mechanism, lepton_asymmetry, omega_b_h2, tau_n
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def sbl_anomaly_check() -> str:
    """What does the BEST experiment's sterile neutrino imply for BBN and Planck?"""
    import json
    result = check_sbl_anomaly()
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def xray_signal(
    m_s_eV: float = 7.1e3,
    sin2_2theta: float = 7e-11,
    target: str = "Perseus",
) -> str:
    """Predict X-ray decay line from sterile neutrino DM in an astrophysical target.

    Parameters:
        m_s_eV: sterile neutrino mass in eV (default: 7100 for the 3.5 keV line)
        sin2_2theta: mixing parameter (default: 7e-11)
        target: one of Perseus, M31, Milky_Way_center, Coma, stacked_galaxies
    """
    import json
    result = predict_xray_signal(m_s_eV, sin2_2theta, target)
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def scan_parameter_space(
    m_min_eV: float = 1e2,
    m_max_eV: float = 1e5,
    theta_min: float = 1e-15,
    theta_max: float = 1e-5,
    n_points: int = 60,
    mechanism: str = "DW",
) -> str:
    """Scan the (m_s, sin²2θ) plane evaluating all constraints. Saves CSV and NPZ."""
    import json
    result = scan_constraints(
        output_dir="output",
        m_min_eV=m_min_eV,
        m_max_eV=m_max_eV,
        theta_min=theta_min,
        theta_max=theta_max,
        n_points=n_points,
        mechanism=mechanism,
    )
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def plot_exclusion_contour(scan_file: str = "output/scan_DW.npz") -> str:
    """Plot the exclusion contour from a parameter space scan. Returns path to PNG."""
    import json
    result = plot_exclusion(scan_file, "output")
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def plot_bbn_neff(
    neff_min: float = 2.0,
    neff_max: float = 5.0,
    n_points: int = 200,
) -> str:
    """Plot Y_p and D/H as functions of N_eff with observational data bands."""
    import json
    result = plot_bbn_vs_neff("output", neff_min, neff_max, n_points)
    return json.dumps(result, indent=2, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio")


# ── Dispatch tools ──────────────────────────────────────────────────

mcp.tool()(set_dispatch)
mcp.tool()(get_dispatch)
mcp.tool()(auth_status)
mcp.tool()(export_dispatch_pack)
