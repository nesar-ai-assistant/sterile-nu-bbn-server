"""Tests for the sterile neutrino physics engine and MCP tools."""
import numpy as np
import pytest

from tools.physics import (
    bbn_dh,
    bbn_yp,
    chi2_bbn,
    dw_delta_neff,
    dw_relic_density,
    free_streaming_length_kpc,
    radiative_decay_rate,
    radiative_lifetime,
    sf_relic_density,
    xray_flux_from_halo,
    xray_line_energy_keV,
)
from tools.physics import bbn_li7
from tools.sterile_tools import (
    plot_exclusion,
    predict_grid,
    check_sbl_anomaly,
    describe_sterile_nu_tools,
    predict_from_particle_params,
    predict_xray_signal,
    scan_constraints,
    plot_bbn_vs_neff,
)


# ═══════════════════════════════════════════════════════════════════
# Physics engine tests
# ═══════════════════════════════════════════════════════════════════

class TestBBN:
    """BBN parameterisation should match known values."""

    def test_standard_yp(self):
        """SM Y_p = 0.2469 (PRyMordial, N_eff = 3.044, Planck omega_b)."""
        yp = bbn_yp(3.044)
        assert abs(yp - 0.2469) < 0.0005

    def test_standard_dh(self):
        """SM D/H x 10^5 = 2.445 (PRyMordial)."""
        dh = bbn_dh(3.044)
        assert abs(dh - 2.445) < 0.01

    def test_slopes_match_prymordial(self):
        """Response to extra radiation and tau_n, from PRyMordial runs:
        dYp/dNeff = 0.0126, d(D/H)/dNeff = 0.33, dYp/dtau_n = +2.1e-4 /s.
        (The original fit had 0.0016, 0.18 and a wrong-sign tau_n term.)"""
        assert bbn_yp(4.044) - bbn_yp(3.044) == pytest.approx(0.0126, abs=0.0004)
        assert bbn_dh(4.044) - bbn_dh(3.044) == pytest.approx(0.332, abs=0.01)
        assert bbn_yp(3.044, tau_n=888.4) - bbn_yp(3.044) == pytest.approx(0.0021, abs=0.0002)

    def test_independent_qke_point(self):
        """PRyMordial-nu-sterile QKE run (Delta m^2 = 1 eV^2, 1e-3 mixing):
        Delta N_eff = 0.930 -> Delta Yp = 0.01174, Delta D/H = 0.309."""
        assert bbn_yp(3.044 + 0.93) - bbn_yp(3.044) == pytest.approx(0.01174, rel=0.03)
        assert bbn_dh(3.044 + 0.93) - bbn_dh(3.044) == pytest.approx(0.309, rel=0.03)

    def test_li7_decreases_with_neff(self):
        assert bbn_li7(4.0) < bbn_li7(3.0)

    def test_delta_chi2_limit(self):
        """BBN alone excludes Delta N_eff ~ 0.55 at 95% (Delta chi^2 vs SM)."""
        assert chi2_bbn(3.044)["delta_chi2_vs_sm"] == 0.0
        assert not chi2_bbn(3.044 + 0.5)["excluded_95"]
        assert chi2_bbn(3.044 + 0.6)["excluded_95"]

    def test_yp_increases_with_neff(self):
        """More radiation → higher n/p freeze-out → more He-4."""
        assert bbn_yp(4.0) > bbn_yp(3.0)

    def test_dh_increases_with_neff(self):
        """More radiation → faster expansion → more D survives."""
        assert bbn_dh(4.0) > bbn_dh(3.0)

    def test_chi2_sm_is_small(self):
        """SM predictions should be consistent with observations."""
        result = chi2_bbn(3.044)
        assert result["chi2_total"] < 6.18  # 2σ at 2 dof
        assert result["consistent_2sigma"] is True


class TestDWProduction:
    """Dodelson-Widrow production tests."""

    def test_zero_mixing(self):
        assert dw_delta_neff(1e3, 0.0) == 0.0
        assert dw_relic_density(1e3, 0.0) == 0.0

    def test_full_thermalisation_ev_scale(self):
        """eV-scale with large mixing → ΔN_eff = 1."""
        dn = dw_delta_neff(1.0, 0.1)
        assert dn == pytest.approx(1.0, abs=0.01)

    def test_relic_density_scaling(self):
        """Ω_s h² should increase with sin²2θ."""
        o1 = dw_relic_density(3e3, 1e-10)
        o2 = dw_relic_density(3e3, 1e-9)
        assert o2 > o1

    def test_kev_delta_neff_consistent_with_relic_density(self):
        """A relativistic-at-BBN population with Omega h^2 has
        Delta N_eff = Omega h^2 * 93.14 eV / m_s (was 2-65x off)."""
        for m, s22 in [(3e3, 1e-10), (7.1e3, 7e-11), (5e4, 1e-12)]:
            assert dw_delta_neff(m, s22) == pytest.approx(
                dw_relic_density(m, s22) * 93.14 / m, rel=1e-9)

    def test_relic_density_all_dm(self):
        """For 3 keV, sin²2θ ≈ 3×10^-9 should give Ω ≈ 0.3."""
        o = dw_relic_density(3e3, 3e-9)
        assert 0.1 < o < 1.0


class TestSFProduction:
    """Shi-Fuller resonant production tests."""

    def test_sf_much_lower_mixing(self):
        """SF gives correct relic for much smaller mixing than DW."""
        # For 7 keV DM, SF needs sin²2θ ~ 2×10^-13 with L ~ 10^-3
        # At these params Ω ≈ 1.2 — correct order of magnitude;
        # the exact DM abundance match requires tuning L precisely
        o_sf = sf_relic_density(7e3, 2e-13, lepton_asymmetry=1e-3)
        assert 0.05 < o_sf < 5.0
        # Crucially, DW at the same mixing gives negligible relic
        o_dw = dw_relic_density(7e3, 2e-13)
        assert o_sf > o_dw * 100  # SF dominates by orders of magnitude

    def test_sf_zero_asymmetry(self):
        assert sf_relic_density(7e3, 1e-10, lepton_asymmetry=0.0) == 0.0


class TestXray:
    """X-ray decay predictions."""

    def test_line_energy(self):
        """7.1 keV sterile ν → 3.55 keV photon."""
        assert xray_line_energy_keV(7.1e3) == pytest.approx(3.55, abs=0.01)

    def test_decay_rate_scaling(self):
        """Γ ∝ m_s^5 sin²2θ."""
        g1 = radiative_decay_rate(7e3, 1e-10)
        g2 = radiative_decay_rate(14e3, 1e-10)
        # Factor of 2^5 = 32 in mass
        assert g2 / g1 == pytest.approx(32.0, rel=0.1)

    def test_decay_rate_absolute(self):
        """Gamma = 9 alpha G_F^2 sin^2 2theta m^5 / (1024 pi^4); the old
        normalisation was 1014x too fast."""
        alpha, GF, hbar = 1 / 137.036, 1.1663788e-23, 6.582119569e-16
        exact = 9 * alpha * GF**2 * 1e-10 * (1e3) ** 5 / (1024 * np.pi**4) / hbar
        assert radiative_decay_rate(1e3, 1e-10) == pytest.approx(exact, rel=0.01)

    def test_lifetime_inverse(self):
        rate = radiative_decay_rate(7e3, 7e-11)
        tau = radiative_lifetime(7e3, 7e-11)
        assert rate * tau == pytest.approx(1.0, rel=1e-6)

    def test_flux_positive(self):
        flux = xray_flux_from_halo(7e3, 7e-11, 1e14, 100.0)
        assert flux > 0

    def test_flux_decreases_with_distance(self):
        f1 = xray_flux_from_halo(7e3, 7e-11, 1e14, 50.0)
        f2 = xray_flux_from_halo(7e3, 7e-11, 1e14, 100.0)
        assert f1 > f2


class TestFreeStreaming:
    """Structure formation constraints."""

    def test_heavier_is_colder(self):
        """Higher mass → shorter free-streaming → colder DM."""
        l1 = free_streaming_length_kpc(3e3)
        l2 = free_streaming_length_kpc(30e3)
        assert l2 < l1

    def test_kev_scale_is_warm(self):
        """keV-scale sterile ν is warm DM (λ_fs > 10 kpc)."""
        lfs = free_streaming_length_kpc(3e3)
        assert lfs > 10


# ═══════════════════════════════════════════════════════════════════
# MCP tool tests
# ═══════════════════════════════════════════════════════════════════

class TestTools:
    """MCP tool integration tests."""

    def test_describe(self):
        result = describe_sterile_nu_tools()
        assert result["status"] == "ok"
        assert "observations" in result["metadata"]

    def test_predict_ev_scale(self):
        """eV-scale prediction for BEST-like parameters."""
        result = predict_from_particle_params(1.14, 0.42, mechanism="DW")
        assert result["status"] == "ok"
        meta = result["metadata"]
        assert meta["production"]["delta_neff"] == pytest.approx(1.0, abs=0.01)
        assert meta["bbn"]["Yp"] > 0.24

    def test_predict_kev_dm(self):
        """keV-scale DM candidate: 7.1 keV -> 3.55 keV photon."""
        result = predict_from_particle_params(7.1e3, 7e-11, mechanism="DW")
        assert result["status"] == "ok"
        meta = result["metadata"]
        assert meta["xray"]["E_gamma_keV"] == pytest.approx(3.55, abs=0.01)

    def test_sbl_anomaly(self):
        result = check_sbl_anomaly()
        assert result["status"] == "ok"
        assert "tension_with_planck_sigma" in result["metadata"]["sbl_assessment"]

    def test_xray_prediction(self):
        result = predict_xray_signal(7.1e3, 7e-11, "Perseus")
        assert result["status"] == "ok"
        assert result["metadata"]["flux_ph_s_cm2"] > 0

    def test_xray_all_targets_and_bad_target(self):
        result = predict_xray_signal(7.1e3, 7e-11, "all")
        assert set(result["metadata"]["flux_ph_s_cm2_by_target"]) == {
            "Perseus", "M31", "Milky_Way_center", "Coma", "stacked_galaxies"}
        with pytest.raises(ValueError, match="Unknown target"):
            predict_xray_signal(7.1e3, 7e-11, "Virgo")  # used to fall back silently

    def test_mev_masses_flagged(self):
        meta = predict_from_particle_params(5e8, 1e-6)["metadata"]
        assert meta["regime"]["bbn_treatment_valid"] is False
        assert predict_from_particle_params(7.1e3, 7e-11)["metadata"]["regime"]["bbn_treatment_valid"]

    def test_grid(self, tmp_path):
        r = predict_grid([1.14, 7100, 5e7], [0.42, 7e-11, 1e-12], "zip",
                         output_dir=str(tmp_path))
        rows = r["metadata"]["rows"]
        assert len(rows) == 3 and rows[0]["bbn_excluded_95"]
        assert r["metadata"]["regimes_outside_validity"] == ["MeV+ (heavy neutral lepton)"]
        r = predict_grid([1e3, 1e4], [1e-12, 1e-10, 1e-8], output_dir=str(tmp_path))
        assert r["metadata"]["n_rows"] == 6
        with pytest.raises(ValueError):
            predict_grid([1.0], [2.0], output_dir=str(tmp_path))

    def test_scan(self, tmp_path):
        result = scan_constraints(
            output_dir=str(tmp_path),
            n_points=5,
        )
        assert result["status"] == "ok"
        assert len(result["files"]) == 2
        assert "bbn_max_allowed_sin2_2theta_by_mass_eV" in result["metadata"]

    def test_scans_do_not_overwrite_and_plot_matches(self, tmp_path):
        a = scan_constraints(output_dir=str(tmp_path), n_points=12)
        b = scan_constraints(output_dir=str(tmp_path), n_points=12, m_max_eV=1e9)
        assert a["files"][1] != b["files"][1]
        pa = plot_exclusion(a["files"][1], str(tmp_path))
        pb = plot_exclusion(b["files"][1], str(tmp_path), save_pdf=True)
        assert pa["files"][0] != pb["files"][0]
        assert pb["files"][1].endswith(".pdf")
        assert pb["metadata"]["hatched_invalid_region"]
        # default-range scan keeps the notebook-compatible name
        c = scan_constraints(output_dir=str(tmp_path))
        assert c["files"][1].endswith("scan_DW.npz")

    def test_plot_bbn_neff(self, tmp_path):
        result = plot_bbn_vs_neff(output_dir=str(tmp_path), n_points=20)
        assert result["status"] == "ok"
        assert len(result["files"]) == 1
        assert result["metadata"]["bbn_delta_neff_limit_95"] == pytest.approx(0.55, abs=0.03)


# ═══════════════════════════════════════════════════════════════════
# MCP surface: names, skills
# ═══════════════════════════════════════════════════════════════════

EXPECTED_TOOLS = {
    "describe_sterile_nu_tools", "predict_sterile_nu", "predict_sterile_nu_grid",
    "check_sterile_nu_sbl_anomaly", "predict_sterile_nu_xray_line",
    "scan_sterile_nu_parameters", "plot_sterile_nu_exclusion",
    "plot_sterile_nu_bbn_impact", "list_sterile_nu_skills", "load_sterile_nu_skill",
}


def _mcp_tool_names():
    import asyncio
    from mcp_server import mcp
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_mcp_tool_names_are_domain_specific():
    names = _mcp_tool_names()
    assert names == EXPECTED_TOOLS
    # every name carries the server's domain token
    assert all("sterile_nu" in n for n in names)


def _unknown_tool_refs(text, names):
    import re
    called = set(re.findall(r"`([a-z_]+)(?:\(|`)", text))
    called |= set(re.findall(r"\b([a-z]+_[a-z_]+)\(", text))
    tool_like = {c for c in called if "sterile_nu" in c
                 or c.startswith(("predict_", "scan_", "plot_", "check_", "list_", "load_", "describe_"))}
    return tool_like - names


def test_skills_reference_real_tools():
    from tools.skills import skill_files
    names = _mcp_tool_names()
    files = skill_files()
    assert len(files) >= 2
    for path in files:
        unknown = _unknown_tool_refs(path.read_text(), names)
        assert not unknown, f"{path.name} names non-tools: {sorted(unknown)}"
    # the check must catch stale names (mutation check)
    assert _unknown_tool_refs("call `scan_parameter_space` then plot_bbn_neff(x)", names) == {
        "scan_parameter_space", "plot_bbn_neff"}


def test_skill_tools_and_prompts():
    import asyncio, json
    from mcp_server import load_sterile_nu_skill, list_sterile_nu_skills, mcp
    index = json.loads(list_sterile_nu_skills())["metadata"]["skills"]
    assert {"sterile-nu-bbn-constraints", "kev-sterile-dm-xray"} <= set(index)
    assert all(index.values())
    text = json.loads(load_sterile_nu_skill("kev-sterile-dm-xray"))["metadata"]["instructions"]
    assert "predict_sterile_nu_xray_line" in text
    prompts = {p.name for p in asyncio.run(mcp.list_prompts())}
    assert set(index) <= prompts


def test_plot_exclusion_accepts_scan_csv(tmp_path):
    """The scan lists its CSV first; plotting from it must work too."""
    from tools.sterile_tools import plot_exclusion, scan_constraints
    scan = scan_constraints(output_dir=str(tmp_path), n_points=12)
    csv = next(f for f in scan["files"] if f.endswith(".csv"))
    from_csv = plot_exclusion(scan_file=csv, output_dir=str(tmp_path))
    assert from_csv["files"][0].endswith(".png")
