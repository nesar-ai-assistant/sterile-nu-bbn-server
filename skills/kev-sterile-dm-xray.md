---
name: kev-sterile-dm-xray
description: keV sterile-neutrino dark matter — the 3.5 keV line, radiative-decay X-ray flux across targets (Perseus, M31, Milky Way, Coma, stacks), NuSTAR/XMM limits, relic density and overclosure, Lyman-alpha and free-streaming bounds
---

# keV sterile-neutrino dark matter recipe

A keV sterile neutrino decays radiatively nu_s -> nu + gamma, giving an
X-ray line at E = m_s / 2 with rate Gamma = 1.36e-32 s^-1 (sin^2 2theta /
1e-10)(m_s / keV)^5. This recipe checks a candidate against X-ray,
abundance and structure-formation constraints.

## 1. One candidate, all targets — one call

`predict_sterile_nu_xray_line(m_s_eV=..., sin2_2theta=..., target="all")`
returns the line energy, lifetime, the whole-halo flux for every target
ranked by brightness, and which tabulated limits exclude it. Do not loop
over targets.

## 2. Is it the dark matter?

`predict_sterile_nu` (or `predict_sterile_nu_grid` for several candidates)
gives Omega_s h^2 (DW, or SF with a lepton asymmetry), `overcloses`
(Omega_s h^2 > 0.12), Delta N_eff (tiny for keV DM — BBN does not constrain
it), and the free-streaming length. For DW production, the mixing needed
for the full relic density at m_s ~ 7 keV is already X-ray excluded — say
so; resonant (SF) production is the surviving option.

## 3. The plane

`scan_sterile_nu_parameters(m_min_eV=1e3, m_max_eV=1e5, theta_min=1e-14,
theta_max=1e-6)` then `plot_sterile_nu_exclusion(scan_file=<its NPZ>)`:
overclosure region, Omega_DM line, NuSTAR/XMM limits, Lyman-alpha bound,
3.5 keV point.

## Caveats you must state

- **Fluxes are whole-halo** with the sterile neutrino as all of the dark
  matter: no field-of-view cut, no density-profile integration. Compare to
  a measured line flux only as an order-of-magnitude check.
- **X-ray limits are tabulated points** (NuSTAR M31, XMM blank sky);
  "excluded_by" checks limits within 30% in mass only.
- **Lyman-alpha**: 5.3 keV is the THERMAL-relic WDM bound; the equivalent
  bound on a DW sterile neutrino mass is several times higher.
- **The 3.5 keV line is debated** (Dessert+ 2020 blank-sky exclusion).
