from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "paper" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def plot_com_trajectory() -> None:
    traj = pd.read_csv(RESULTS / "gravity_analysis" / "gravity_energy_timeseries.csv")
    with open(RESULTS / "movement_analysis" / "movement_analysis.json", "r", encoding="utf-8") as f:
        movement = json.load(f)

    turning_points = movement["segmentation"]["turning_points"]
    highlight = [p for p in turning_points if p.get("kind") in {"high", "low"}]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=220)
    ax.plot(
        traj["time"],
        traj["com_y_m"],
        color="#1f4e79",
        linewidth=2.2,
        label="COM height",
    )

    for point in highlight:
        color = "#d7263d" if point["kind"] == "high" else "#f28e2b"
        ax.scatter(
            point["time"],
            point["value"],
            s=38,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )
        ax.annotate(
            point["kind"].title(),
            (point["time"], point["value"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="black",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("COM height (m)")
    ax.set_title("Vertical center-of-mass trajectory")
    ax.grid(True, alpha=0.25)
    ax.margins(x=0.02)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig4_com_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gravity_pe() -> None:
    traj = pd.read_csv(RESULTS / "gravity_analysis" / "gravity_energy_timeseries.csv")
    cols = {
        "microgravity": "potential_energy_microgravity_j",
        "moon": "potential_energy_moon_j",
        "mars": "potential_energy_mars_j",
        "earth": "potential_energy_earth_j",
    }
    colors = {
        "microgravity": "#5b5b5b",
        "moon": "#2f6fed",
        "mars": "#0b9f6e",
        "earth": "#d62828",
    }

    fig, ax = plt.subplots(figsize=(9, 5), dpi=220)
    for name, column in cols.items():
        ax.plot(
            traj["time"],
            traj[column],
            color=colors[name],
            linewidth=2.0,
            label=name.replace("_", " ").title(),
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative potential energy (J)")
    ax.set_title("Relative gravitational potential energy vs time")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.margins(x=0.02)
    plt.tight_layout()
    fig.savefig(OUTDIR / "fig5_gravity_pe.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_com_trajectory()
    plot_gravity_pe()
    print(f"Saved figures to: {OUTDIR}")
