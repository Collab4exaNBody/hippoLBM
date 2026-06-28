#!/usr/bin/env python3
"""AI GENERATED FILE
Plot a hippoLBM velocity profile CSV file (position avg min max).

Usage:
  plot_profile.py path/to/profile_0000000160.csv [-o output_dir]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_profile(csv_path):
    data = np.loadtxt(csv_path, skiprows=1)
    position, avg, vmin, vmax = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return position, avg, vmin, vmax


def plot_velocity_with_bounds(position, avg, vmin, vmax, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    lower_err = avg - vmin
    upper_err = vmax - avg
    ax.errorbar(position, avg, yerr=[lower_err, upper_err], fmt="-o", markersize=3,
                capsize=3, ecolor="gray", label="avg (min/max bounds)")
    ax.set_xlabel("position")
    ax.set_ylabel("velocity")
    ax.set_title("Velocity profile with min/max bounds")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_avg_min_max(position, avg, vmin, vmax, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(position, avg, color="tab:blue")
    axes[0].set_ylabel("avg")
    axes[0].grid(True)

    axes[1].plot(position, vmin, color="tab:orange")
    axes[1].set_ylabel("min")
    axes[1].grid(True)

    axes[2].plot(position, vmax, color="tab:green")
    axes[2].set_ylabel("max")
    axes[2].set_xlabel("position")
    axes[2].grid(True)

    fig.suptitle("avg / min / max velocity profiles")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", help="Path to the profile CSV file (e.g. PoiseuilleTestDir/Profile/profile_0000000160.csv)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Directory where the PNG files are written (default: same directory as the CSV file)")
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.csv_file))[0]
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv_file))
    os.makedirs(output_dir, exist_ok=True)

    position, avg, vmin, vmax = load_profile(args.csv_file)

    bounds_png = os.path.join(output_dir, base_name + "_velocity_bounds.png")
    plot_velocity_with_bounds(position, avg, vmin, vmax, bounds_png)

    avg_min_max_png = os.path.join(output_dir, base_name + "_avg_min_max.png")
    plot_avg_min_max(position, avg, vmin, vmax, avg_min_max_png)

    print(f"wrote {bounds_png}")
    print(f"wrote {avg_min_max_png}")


if __name__ == "__main__":
    main()
