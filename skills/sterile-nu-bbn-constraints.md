---
name: sterile-nu-bbn-constraints
description: Constrain sterile neutrinos with BBN and N_eff — Delta N_eff from (m_s, sin^2 2theta), Yp/D/H shifts, BBN exclusion contours in the (m_s, mixing) plane, the BEST short-baseline anomaly vs BBN and Planck, and which mass ranges the Delta-N_eff treatment is valid for
---

# Sterile neutrinos x BBN recipe

What this server computes: the extra radiation Delta N_eff a sterile
neutrino produces (Dodelson-Widrow or Shi-Fuller), the light-element shifts
that follows (fit to the PRyMordial network, <0.3% inside its validity
box), and a BBN verdict as Delta chi^2 vs the Standard Model (excluded if
> 3.84, 95% CL, one degree of freedom — the only BBN effect here is one
number, Delta N_eff).

## 0. Check the mass regime FIRST

Every result carries `regime` / `bbn_treatment_valid`:

| m_s | regime | BBN here is... |
|---|---|---|
| < 10 eV | short-baseline | valid; partial thermalisation is a crude fit |
| 10 eV - 100 keV | keV warm DM | valid (relativistic at BBN) |
| 100 keV - 1 MeV | transitional | approximate |
| >= 1 MeV | heavy neutral lepton | NOT a constraint: decays during BBN (photodissociation, hadronic injection, entropy release) are not modeled |

For MeV+ masses say plainly that this server cannot constrain them, and
that the dominant BBN bound comes from decay injection (e.g. ACROPOLIS-type
photodissociation, Chen & Zhang arXiv:2410.07343). Never present a MeV+
number from this server as an exclusion.

## 1. Benchmark points and sweeps — one call

Use `predict_sterile_nu_grid`, not repeated `predict_sterile_nu` calls:

- benchmarks: `masses_eV=[1.14, 7100, 50000]`, `sin2_2thetas=[0.42, 7e-11,
  1e-12]`, `combine="zip"`
- a Delta N_eff-vs-mixing curve at fixed mass: `masses_eV=[1.0]`,
  `sin2_2thetas=[1e-5, 1e-4, 1e-3, 1e-2, 0.1]`

Quote `delta_neff`, `delta_chi2_bbn`, `bbn_excluded_95`, `overcloses` from
the inline rows; the CSV has every column (Yp, D/H, Li7, lifetime, ...).

## 2. Exclusion plane

1. `scan_sterile_nu_parameters` over the range of interest (stay below
   ~1e5 eV unless you want the invalid region hatched). Its metadata gives
   `bbn_max_allowed_sin2_2theta_by_mass_eV` — quote those numbers directly.
2. `plot_sterile_nu_exclusion(scan_file=<the NPZ just returned>)` — always
   pass the file from THIS scan (each range has its own file name).
3. `plot_sterile_nu_bbn_impact` shows Yp and D/H vs N_eff with the data
   bands and the BBN 95% limit on Delta N_eff (also in its metadata).

## 3. BEST anomaly

`check_sterile_nu_sbl_anomaly` gives the BEST best fit's Delta N_eff, the
Planck tension in sigma, and the BBN Delta chi^2. A fully thermalised eV
sterile (Delta N_eff ~ 1) is strongly disfavoured by both unless
production is suppressed (e.g. large lepton asymmetry, secret
interactions) — none of which this server models.

## Caveats you must state

- **BBN is a fit, not a network run.** For full-network abundances at a
  given N_eff (or other omega_b, tau_n), use a dedicated BBN abundance
  server if connected (e.g. its compute_bbn_abundances or scan_bbn_neff).
- **eV partial thermalisation is crude**: a full QKE (PRyMordial-nu-
  sterile) gives Delta N_eff = 0.93 at Delta m^2 = 1 eV^2, sin^2 2theta =
  1e-3 where this server gives 0.40 — eV-scale limits here are
  conservative.
- **The 10 eV seam**: below 10 eV Delta N_eff comes from the eV fit, above
  it from the DW relic density; the BBN limit on sin^2 2theta jumps by ~30x
  across m_s = 10 eV. Do not quote limits right at the seam.
- **Shi-Fuller Delta N_eff is a heuristic** (0.1 x DW); its relic density
  is a fitting formula in the lepton asymmetry.
- **D/H tension**: the SM itself sits ~1.6 sigma low in D/H; that is why
  the verdict uses Delta chi^2 vs the SM, never the absolute chi^2.
