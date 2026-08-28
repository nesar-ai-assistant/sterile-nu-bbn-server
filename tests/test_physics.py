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
from tools.sterile_tools import (
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
        """SM prediction Y_p ≈ 0.2485 for N_eff = 3.044."""
        yp = bbn_yp(3.044)
        assert abs(yp - 0.2485) < 0.001

    def test_standard_dh(self):
        """SM prediction D/H × 10^5 ≈ 2.57."""
        dh = bbn_dh(3.044)
        assert abs(dh - 2.57) < 0.05

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

    def test_scan(self, tmp_path):
        result = scan_constraints(
            output_dir=str(tmp_path),
            n_points=5,
        )
        assert result["status"] == "ok"
        assert len(result["files"]) == 2

    def test_plot_bbn_neff(self, tmp_path):
        result = plot_bbn_vs_neff(output_dir=str(tmp_path), n_points=20)
        assert result["status"] == "ok"
        assert len(result["files"]) == 1
