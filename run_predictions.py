#!/usr/bin/env python3
"""Run predictions and generate output plots."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tools.sterile_tools import (
    check_sbl_anomaly,
    predict_from_particle_params,
    predict_xray_signal,
    scan_constraints,
    plot_exclusion,
    plot_bbn_vs_neff,
)

print("=" * 60)
print("1. BEST ANOMALY -> BBN IMPACT")
print("=" * 60)
r = check_sbl_anomaly()
print(r["message"])
print()

print("=" * 60)
print("2. 7.1 keV DM CANDIDATE (3.5 keV line)")
print("=" * 60)
r2 = predict_from_particle_params(7.1e3, 7e-11, mechanism="DW")
print(r2["message"])
print()

print("=" * 60)
print("3. X-RAY SIGNAL IN PERSEUS CLUSTER")
print("=" * 60)
r3 = predict_xray_signal(7.1e3, 7e-11, "Perseus")
print(r3["message"])
print()

print("=" * 60)
print("4. PARAMETER SPACE SCAN")
print("=" * 60)
r4 = scan_constraints(output_dir="output", n_points=60)
print(r4["message"])
print()

print("=" * 60)
print("5. GENERATING PLOTS")
print("=" * 60)
r5 = plot_exclusion("output/scan_DW.npz", "output")
print(r5["message"])

r6 = plot_bbn_vs_neff("output")
print(r6["message"])

print("\nDone!")
