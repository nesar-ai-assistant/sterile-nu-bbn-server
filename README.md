# sterile-nu-bbn-server

An MCP server that **bridges sterile neutrino particle physics with Big Bang
Nucleosynthesis cosmology** — the first tool that goes end-to-end from
particle parameters to cosmological observables in a single call.

Built as a companion to [`bbn-mcp-server`](https://github.com/nesar/bbn-mcp-server),
extending the BBN framework into the BSM particle physics domain.

## The science

Sterile neutrinos are hypothetical neutral fermions that mix with active
neutrinos.  They appear in many extensions of the Standard Model and are
candidates for dark matter (keV-scale) or the source of short-baseline
oscillation anomalies (eV-scale).  Their existence has direct, quantitative
consequences for:

| Scale | Observable | Data |
|---|---|---|
| **Particle** | Mixing angle sin²2θ, mass m_s | SBL experiments (BEST, STEREO), reactor anomalies |
| **Nuclear / BBN** | Primordial He-4 (Y_p), D/H, ⁷Li/H | Quasar absorption spectra, H II regions |
| **CMB** | N_eff (number of relativistic species) | Planck 2018, ACT, SPT; future CMB-S4 |
| **X-ray** | Decay line ν_s → ν_a + γ | NuSTAR, XMM-Newton, XRISM, Chandra |
| **LSS** | Free-streaming suppression | Lyman-α forest (SDSS, DESI) |

This server computes the **entire chain**: given (m_s, sin²2θ, production
mechanism), it predicts ΔN_eff, BBN abundances, X-ray line flux, and
free-streaming length — then checks all predictions against actual
observational data.

## Tools

MCP tool names all carry a `sterile_nu` token so agents never confuse them
with a BBN abundance server's or a client harness's tools.

| MCP tool | What it does |
|---|---|
| `describe_sterile_nu_tools` | Capabilities, physics, validity range, observational data — call first |
| `predict_sterile_nu` | **The bridge** for one point: ΔN_eff, Ω_s h², Y_p/D/H/Li7, BBN Δχ² vs SM, X-ray line, free streaming, regime flag |
| `predict_sterile_nu_grid` | Same for many points in one call (product or zip of mass/mixing lists) → compact table + CSV |
| `check_sterile_nu_sbl_anomaly` | What BEST's best-fit sterile ν means for BBN and Planck N_eff |
| `predict_sterile_nu_xray_line` | X-ray line flux from DM decay; `target="all"` compares Perseus, M31, MW, Coma, stacks |
| `scan_sterile_nu_parameters` | Scan the (m_s, sin²2θ) plane → CSV/NPZ, BBN-allowed mixing per mass |
| `plot_sterile_nu_exclusion` | Exclusion plane: BBN, overclosure, X-ray, Ly-α, 3.5 keV, invalid-regime hatching |
| `plot_sterile_nu_bbn_impact` | Y_p, D/H vs N_eff with observed bands and the BBN 95% ΔN_eff limit |
| `list_sterile_nu_skills` / `load_sterile_nu_skill` | Workflow recipes in `skills/` (also exposed as MCP prompts) |

Python functions behind them (`predict_from_particle_params`,
`scan_constraints`, `plot_exclusion`, `plot_bbn_vs_neff`, ...) keep their
names for notebooks and scripts. Figures use a shared publication style
(`tools/plotting.py`: STIX mathtext, no LaTeX needed, colorblind-safe
palette); every plot tool accepts `save_pdf`.

## Physics and validity

- **BBN**: Y_p, D/H, ⁷Li/H are fits to the full PRyMordial network over
  N_eff 2.04–6.04, ω_b ±15%, τ_n ±12 s (max deviation 0.11%, 0.26%, 0.14%).
  Exclusion is Δχ² vs the SM > 3.84 (95%, 1 dof): the SM itself carries
  χ² ≈ 2.8 from the D/H tension, which says nothing about sterile ν.
  BBN alone gives ΔN_eff < 0.55.
- **Mass regimes**: valid below ~100 keV (relativistic at BBN). At ≥ 1 MeV
  the particle decays during BBN and decay-product injection is **not
  modeled** — results there are flagged and are not constraints.
- **Known approximations**: eV partial thermalisation is a crude fit (a
  QKE run gives ΔN_eff = 0.93 at Δm² = 1 eV², sin²2θ = 1e-3 vs 0.40 here);
  Shi–Fuller ΔN_eff is a heuristic; X-ray fluxes are whole-halo.

## Run the server

```bash
python -m mcp_server
```

## Questions to ask an agent

> "If the BEST experiment's sterile neutrino is real, what happens to
> primordial helium and deuterium?"

> "Can a 7 keV sterile neutrino be all the dark matter, and would
> XRISM see its X-ray decay line in the Perseus cluster?"

> "Scan the parameter space and show me where BBN, X-ray limits,
> and structure formation all agree."

## Physics references

- Dodelson & Widrow, PRL 72, 17 (1994) — non-resonant production
- Shi & Fuller, PRL 82, 2832 (1999) — resonant production
- Pal & Wolfenstein, PRD 25, 766 (1982) — radiative decay rate
- Burns, Tait & Valli, EPJC 84, 86 (2024) — PRyMordial (BBN calibration)
- Pitrou, Coc, Uzan & Vangioni, PhysRep 754, 1 (2018) — BBN review
- Roach+ (2020), PRD 101, 103011 — NuSTAR X-ray limits
- Dessert+ (2020), Science 367, 1465 — XMM-Newton blank-sky limits
- Barinov+ (2022), PRL 128, 232501 — BEST experiment
- Aver, Olive & Skillman, JCAP 07 (2021) 029 — primordial He-4
- Cooke, Pettini & Steidel, ApJ 855 (2018) 102 — primordial D/H

## License

MIT
