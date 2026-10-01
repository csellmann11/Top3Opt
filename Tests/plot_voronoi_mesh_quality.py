"""Plot geometric quality only; no density solve is needed.

Run after voronoi_mesh_audit.jl and voronoi_geometric_probe.jl.
"""
from pathlib import Path
import csv
import sys

# Optional project-local plotting dependencies, as used by other diagnostics.
deps = Path(__file__).resolve().parents[1] / "Results" / "plot_deps"
if deps.is_dir():
    sys.path.insert(0, str(deps))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

out = Path(__file__).resolve().parents[1] / "Results" / "voronoi_mesh_audit"
with (out / "front_face_quality.csv").open() as stream:
    rows = list(csv.DictReader(stream))
polygons = [np.array([[float(v) for v in p.split()] for p in row["vertices_xz"].split(";")])
            for row in rows]
quality = np.array([float(row["edge_ratio"]) for row in rows])
worst = int(np.argmax(quality))
x, z = float(rows[worst]["x"]), float(rows[worst]["z"])
fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 1.35]})
for ax in axes:
    collection = PolyCollection(polygons, array=quality, cmap="magma_r",
                                edgecolors="#293342", linewidths=0.25,
                                clim=(1, quality.max()))
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
axes[0].set_xlim(0, 3)
axes[0].set_ylim(0, 1)
axes[0].set_title("MBB Voronoi, refinement level 4 — front surface, geometric quality")
half = 0.22
axes[1].set_xlim(max(0, x-half), min(3, x+half))
axes[1].set_ylim(max(0, z-half), min(1, z+half))
axes[1].set_title(f"Detail at largest edge ratio: {quality[worst]:.2f}, face {rows[worst]['face_id']}")
axes[1].plot(x, z, "o", ms=8, mec="#00b9cd", mfc="none", mew=1.8)
fig.colorbar(collection, ax=axes, label="Longest / shortest edge within a surface cell", shrink=0.7, pad=0.02)
fig.savefig(out / "mesh_quality.png", dpi=180, bbox_inches="tight")
fig.savefig(out / "mesh_quality.svg", bbox_inches="tight")
print(out / "mesh_quality.png")
