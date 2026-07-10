#!/usr/bin/env python3
"""AI GENERATED FILE
Plot ux vs rz from a hippoLBM `plot_line_velocity` CSV file (Poiseuille flow)
and overlay the analytic transient (startup) plane-Poiseuille solution:

  u(z, t) = (4 F L^2) / (nu * pi^3)
            * sum_{k=0}^inf [1 / (2k+1)^3] * sin((2k+1) pi z / L)
                           * (1 - exp(-nu * ((2k+1) pi / L)^2 * t))

where F is the driving body force per unit mass (Fext_x), nu is the
kinematic viscosity, L is the distance between the two no-slip walls, and
t = timestep * dt (timestep is parsed from the file name, e.g.
line_0000000300.csv -> timestep 300). As t -> infinity this relaxes to the
steady-state parabola F/(2 nu) * z * (L - z).

Usage:
  plot_line_poiseuille.py PoiseuilleTestDir/Profile/line_0000000300.csv [more.csv ...]
  plot_line_poiseuille.py PoiseuilleTestDir/Profile   # scans every line_*.csv in the directory
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MAX_LEGEND_ENTRIES = 10

TIMESTEP_RE = re.compile(r"(\d+)(?=\.csv$)")


def resolve_csv_files(paths):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "line_*.csv")))
        else:
            files.extend(glob.glob(path))
    if not files:
        raise FileNotFoundError(f"no line_*.csv files found in: {paths}")
    return sorted(set(files), key=parse_timestep)


def parse_timestep(csv_path):
    match = TIMESTEP_RE.search(os.path.basename(csv_path))
    if not match:
        raise ValueError(f"could not parse timestep from file name: {csv_path}")
    return int(match.group(1))


def load_line(csv_path):
    data = np.loadtxt(csv_path, skiprows=1)
    rz, ux = data[:, 2], data[:, 3]
    return rz, ux


def analytic_poiseuille(z, t, F, nu, L, n_terms=50):
    k = np.arange(n_terms)
    n = 2 * k + 1  # odd modes only
    # shape: (n_terms, len(z))
    modes = np.sin(np.outer(n, np.pi * z / L)) / n[:, None] ** 3
    decay = 1.0 - np.exp(-nu * (n * np.pi / L) ** 2 * t)
    return (4.0 * F * L**2) / (nu * np.pi**3) * np.sum(modes * decay[:, None], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_files", nargs="+",
                        help="Path(s) to line_<timestep>.csv file(s), and/or directories to scan for line_*.csv")
    parser.add_argument("--dt", type=float, default=3.3333333e-03,
                        help="Time step (s), default: 3.3333333e-03 (dx/celerity with dx=0.1/30, celerity=1)")
    parser.add_argument("-F", "--force", type=float, default=9.512485e-05,
                        help="Driving body force per unit mass (Fext_x), default: 9.512485e-05")
    parser.add_argument("--nu", type=float, default=1e-3, help="Kinematic viscosity used in the analytic solution")
    parser.add_argument("-L", "--domain-length", type=float, default=0.1, help="Distance between the two walls")
    parser.add_argument("--n-terms", type=int, default=50, help="Number of Fourier modes used in the analytic sum")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (default: shown alongside the first CSV file)")
    args = parser.parse_args()
    csv_files = resolve_csv_files(args.csv_files)

    fig, ax = plt.subplots(figsize=(8, 6))
    timesteps = [parse_timestep(p) for p in csv_files]
    colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))
    use_per_file_legend = len(csv_files) <= MAX_LEGEND_ENTRIES

    for csv_path, timestep, color in zip(csv_files, timesteps, colors):
        t = timestep * args.dt

        rz, ux = load_line(csv_path)
        sim_label = f"simulation (n={timestep})" if use_per_file_legend else None
        ax.plot(rz, ux, "o", markersize=3, color=color, label=sim_label)

        z_analytic = np.linspace(0.0, args.domain_length, 200)
        u_analytic = analytic_poiseuille(z_analytic, t, args.force, args.nu, args.domain_length, args.n_terms)
        analytic_label = f"analytic (n={timestep}, t={t:.3e}s)" if use_per_file_legend else None
        ax.plot(z_analytic, u_analytic, "-", color=color, label=analytic_label)

    ax.set_xlabel("rz")
    ax.set_ylabel("ux")
    ax.set_title("Poiseuille flow velocity profile: simulation vs analytic")
    ax.grid(True)

    if use_per_file_legend:
        ax.legend()
    else:
        # Too many files for a per-timestep legend: use a colorbar for the
        # timestep and a small proxy legend for marker vs line style instead.
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=min(timesteps), vmax=max(timesteps)))
        fig.colorbar(sm, ax=ax, label="timestep")
        proxies = [
            Line2D([0], [0], marker="o", linestyle="", color="gray", label="simulation"),
            Line2D([0], [0], linestyle="-", color="gray", label="analytic"),
        ]
        ax.legend(handles=proxies)

    fig.tight_layout()

    output_path = args.output or os.path.join(os.path.dirname(os.path.abspath(csv_files[0])), "line_poiseuille.png")
    fig.savefig(output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
