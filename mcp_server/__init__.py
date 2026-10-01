"""MCP server entry point for sterile-nu-bbn-server (MCP SDK v2).

Tool names carry a sterile_nu token so they cannot be confused with other
servers' tools (e.g. a BBN abundance server's scan/plot tools, or a client
harness's describe/load_skill tools).
"""
from __future__ import annotations

import json
from typing import Annotated, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from tools.skills import skill_index, skill_text
from tools.sterile_tools import (
    describe_sterile_nu_tools as _describe,
    predict_from_particle_params,
    predict_grid,
    check_sbl_anomaly,
    predict_xray_signal,
    scan_constraints,
    plot_exclusion,
    plot_bbn_vs_neff,
)

mcp = FastMCP(
    name="sterile-nu-bbn-server",
    instructions=(
        "Sterile-neutrino phenomenology: maps (m_s, sin^2 2theta, production "
        "mechanism) to what the particle PRODUCES — Delta N_eff, relic "
        "density, X-ray decay line, free streaming — and to the BBN and "
        "X-ray constraints that follow. BBN abundances come from a fit to "
        "the PRyMordial network; for full-network abundances at a given "
        "N_eff use a BBN abundance server if one is connected. Valid for "
        "m_s below ~100 keV; MeV+ results are flagged (decay injection is "
        "not modeled). For many points use predict_sterile_nu_grid, not "
        "repeated predict_sterile_nu calls. Recipes: list_sterile_nu_skills "
        "then load_sterile_nu_skill."
    ),
)

Mechanism = Literal["DW", "SF"]
MassEV = Annotated[float, Field(gt=0, le=1e10, description="Sterile neutrino mass in eV (1.14 = BEST; 7100 = 3.5 keV line)")]
Mixing = Annotated[float, Field(gt=0, le=1.0, description="Mixing sin^2(2 theta)")]
OutputDir = Annotated[str, Field(min_length=1, description="Directory for output files (default: the client's project output dir, else ./output)")]


def _json(result: dict) -> str:
    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def describe_sterile_nu_tools() -> str:
    """Describe this sterile-neutrino server: its tools, physics, validity range, and the observational data it uses. Call this first."""
    return _json(_describe())


@mcp.tool()
def predict_sterile_nu(
    m_s_eV: MassEV,
    sin2_2theta: Mixing,
    mechanism: Annotated[Mechanism, Field(description="'DW' (Dodelson-Widrow, non-resonant) or 'SF' (Shi-Fuller, resonant; its Delta N_eff is a heuristic)")] = "DW",
    lepton_asymmetry: Annotated[float, Field(gt=0, le=0.1, description="Lepton asymmetry L (SF only)")] = 1e-3,
    omega_b_h2: Annotated[float, Field(ge=0.019, le=0.0257, description="Baryon density (Planck 2018: 0.02237)")] = 0.02237,
    tau_n: Annotated[float, Field(ge=866.4, le=890.4, description="Neutron lifetime [s]")] = 878.4,
) -> str:
    """Predict everything ONE sterile neutrino (m_s, sin²2θ) produces and its constraints.

    Returns Delta N_eff and relic density, the BBN abundances it implies
    (Yp, D/H, Li7; PRyMordial-calibrated fit) with Delta chi^2 vs the SM and
    a 95% verdict, the radiative-decay X-ray line and lifetime, the
    free-streaming length, and a regime flag — for m_s >= ~1 MeV the BBN
    numbers ignore decay injection and are NOT constraints. For more than
    a couple of points use predict_sterile_nu_grid (one call, one table).
    """
    return _json(predict_from_particle_params(
        m_s_eV, sin2_2theta, mechanism, lepton_asymmetry, omega_b_h2, tau_n))


@mcp.tool()
def predict_sterile_nu_grid(
    masses_eV: Annotated[list[float], Field(min_length=1, max_length=100, description="Masses in eV")],
    sin2_2thetas: Annotated[list[float], Field(min_length=1, max_length=100, description="Mixings sin^2(2 theta)")],
    combine: Annotated[Literal["product", "zip"], Field(description="'product' = every mass with every mixing; 'zip' = pair the lists element-wise (benchmark points)")] = "product",
    mechanism: Mechanism = "DW",
    lepton_asymmetry: Annotated[float, Field(gt=0, le=0.1)] = 1e-3,
    output_dir: OutputDir = "output",
) -> str:
    """Predict sterile-neutrino observables for MANY points in one call (max 400).

    Same physics as predict_sterile_nu; returns a compact inline table
    (Delta N_eff, Omega_s h^2, BBN Delta chi^2 and verdict, overclosure,
    lifetime, regime) and a CSV with every column. Use for benchmark
    tables, mass or mixing sweeps, and Delta N_eff-vs-parameter curves.
    """
    return _json(predict_grid(masses_eV, sin2_2thetas, combine, mechanism,
                              lepton_asymmetry, output_dir))


@mcp.tool()
def check_sterile_nu_sbl_anomaly() -> str:
    """Assess the BEST short-baseline sterile neutrino (Δm² = 1.3 eV², sin²2θ = 0.42) against BBN and Planck N_eff."""
    return _json(check_sbl_anomaly())


@mcp.tool()
def predict_sterile_nu_xray_line(
    m_s_eV: MassEV = 7.1e3,
    sin2_2theta: Mixing = 7e-11,
    target: Annotated[Literal["Perseus", "M31", "Milky_Way_center", "Coma", "stacked_galaxies", "all"], Field(description="Astrophysical target, or 'all' to compare every target in one call")] = "all",
) -> str:
    """Predict the X-ray decay line (E = m_s/2) of sterile-neutrino dark matter.

    Gives the line energy, radiative lifetime, whole-halo photon flux per
    target (assuming the sterile ν is all the dark matter; no field-of-view
    cut), and which tabulated X-ray limits (NuSTAR, XMM) exclude the point.
    Default target='all' returns every target in one call.
    """
    return _json(predict_xray_signal(m_s_eV, sin2_2theta, target))


@mcp.tool()
def scan_sterile_nu_parameters(
    m_min_eV: Annotated[float, Field(gt=0)] = 1e2,
    m_max_eV: Annotated[float, Field(gt=0, le=1e10)] = 1e5,
    theta_min: Annotated[float, Field(gt=0)] = 1e-15,
    theta_max: Annotated[float, Field(gt=0, le=1.0)] = 1e-5,
    n_points: Annotated[int, Field(ge=10, le=150)] = 60,
    mechanism: Mechanism = "DW",
    output_dir: OutputDir = "output",
) -> str:
    """Scan the sterile-neutrino (m_s, sin²2θ) plane for BBN and relic-density constraints.

    Writes CSV + NPZ (pass the NPZ to plot_sterile_nu_exclusion) and returns
    counts of BBN-excluded (Delta chi^2 > 3.84) and overclosing points plus
    the largest BBN-allowed sin^2 2theta at sampled masses. Different
    ranges get different file names. Masses >= ~1 MeV are flagged as
    outside model validity.
    """
    if m_min_eV >= m_max_eV or theta_min >= theta_max:
        raise ValueError("Need m_min_eV < m_max_eV and theta_min < theta_max.")
    return _json(scan_constraints(output_dir, m_min_eV, m_max_eV, theta_min,
                                  theta_max, n_points, mechanism))


@mcp.tool()
def plot_sterile_nu_exclusion(
    scan_file: Annotated[str, Field(min_length=1, description="NPZ path returned by scan_sterile_nu_parameters")],
    output_dir: OutputDir = "output",
    title: Annotated[str | None, Field(description="Optional title (default names the production mechanism)")] = None,
    save_pdf: Annotated[bool, Field(description="Also write a vector PDF")] = False,
) -> str:
    """Plot the sterile-neutrino exclusion plane from a scan file.

    BBN exclusion, overclosure and the Omega_DM line, NuSTAR/XMM X-ray
    limits, the Lyman-alpha thermal-WDM bound, the 3.5 keV line, and
    hatching where the BBN treatment is invalid. Publication style.
    """
    return _json(plot_exclusion(scan_file, output_dir, title, save_pdf))


@mcp.tool()
def plot_sterile_nu_bbn_impact(
    neff_min: Annotated[float, Field(ge=2.0, description="Lower N_eff (fit calibrated for N_eff >= 2.044)")] = 2.0,
    neff_max: Annotated[float, Field(le=6.0, description="Upper N_eff (fit calibrated up to 6.044)")] = 5.0,
    n_points: Annotated[int, Field(ge=20, le=1000)] = 200,
    output_dir: OutputDir = "output",
    save_pdf: Annotated[bool, Field(description="Also write a vector PDF")] = False,
) -> str:
    """Plot how extra radiation from sterile neutrinos shifts Yp and D/H.

    Yp and D/H vs N_eff (PRyMordial-calibrated fit) with observed bands,
    the Planck N_eff band, the SM and +1 thermalised-sterile lines, and the
    95% BBN limit on Delta N_eff (quoted in the result).
    """
    if neff_min >= neff_max:
        raise ValueError("Need neff_min < neff_max.")
    return _json(plot_bbn_vs_neff(output_dir, neff_min, neff_max, n_points,
                                  save_pdf))


@mcp.tool()
def list_sterile_nu_skills() -> str:
    """List this sterile-neutrino server's skills (recipes for multi-step workflows); load one with load_sterile_nu_skill when a task matches."""
    index = skill_index()
    return _json({"status": "ok", "files": [],
                  "message": f"{len(index)} skills. Load one with load_sterile_nu_skill.",
                  "metadata": {"skills": index}})


@mcp.tool()
def load_sterile_nu_skill(
    name: Annotated[str, Field(min_length=1, description="Skill name from list_sterile_nu_skills")],
) -> str:
    """Load the full instructions of one of this sterile-neutrino server's skills; then follow them."""
    text = skill_text(name)
    if text is None:
        raise ValueError(f"No skill named {name!r}. Available: "
                         f"{', '.join(sorted(skill_index()))}")
    return _json({"status": "ok", "files": [],
                  "message": f"Loaded skill '{name}'. Follow these instructions.",
                  "metadata": {"instructions": text}})


def _register_skill_prompts() -> None:
    for skill_name, description in skill_index().items():
        def make(n: str):
            def prompt() -> str:
                return skill_text(n)
            return prompt
        mcp.prompt(name=skill_name, description=description)(make(skill_name))


_register_skill_prompts()


if __name__ == "__main__":
    mcp.run(transport="stdio")
