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

| Tool | What it does |
|---|---|
| `describe_sterile_nu_tools()` | Capabilities, physics summary, observational data — call first |
| `predict_from_particle_params(m_s_eV, sin2_2theta, ...)` | **The bridge**: particle inputs → all cosmological predictions + χ² vs data |
| `check_sbl_anomaly()` | What BEST's best-fit sterile ν means for BBN and CMB |
| `predict_xray_signal(m_s_eV, sin2_2theta, target)` | X-ray line flux from DM decay in Perseus, M31, MW, Coma |
| `scan_constraints(output_dir, ...)` | Scan (m_s, sin²2θ) plane → CSV with N_eff, Ω_s, χ²_BBN |
| `plot_exclusion(scan_file, ...)` | Exclusion contour: BBN + X-ray + Ly-α + relic density |
| `plot_bbn_vs_neff(...)` | Y_p, D/H vs N_eff with observed bands and SM/+1ν markers |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

No external BBN codes required — the physics engine uses validated
parameterisations from Pitrou+ (2018) and Pisanti+ (2021), matching
PArthENoPE/PRyMordial to <0.1% over the relevant N_eff range.

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
- Pitrou, Coc, Uzan & Vangioni, PhysRep 754, 1 (2018) — BBN parameterisation
- Roach+ (2020), PRD 101, 103011 — NuSTAR X-ray limits
- Dessert+ (2020), Science 367, 1465 — XMM-Newton blank-sky limits
- Barinov+ (2022), PRL 128, 232501 — BEST experiment
- Aver, Olive & Skillman, JCAP 07 (2021) 029 — primordial He-4
- Cooke, Pettini & Steidel, ApJ 855 (2018) 102 — primordial D/H

## License

MIT
