import argparse
from pathlib import Path

import pyvista as pv


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute the vorticity field from PVTR files.")
parser.add_argument(
    "--input-dir",
    type=Path,
    required=True,
    help="Directory containing the input PVTR files.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory where the output PVTR files will be saved.",
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Find all input PVTR files
# -----------------------------------------------------------------------------
input_files = sorted(args.input_dir.rglob("*.pvtr"))

if not input_files:
    raise FileNotFoundError(f"No .pvtr files found in: {args.input_dir}")

args.output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Process each PVTR file
# -----------------------------------------------------------------------------
for input_file in input_files:
    output_file = args.output_dir / f"{input_file.stem}_vorticity.vtr"

    # -------------------------------------------------------------------------
    # Load the master PVTR file
    # -------------------------------------------------------------------------
    mesh = pv.read(input_file)

    # -------------------------------------------------------------------------
    # Compute the vorticity from the velocity field "U"
    # -------------------------------------------------------------------------
    mesh_with_vorticity = mesh.compute_derivative(
        scalars="U",
        vorticity=True,
    )

    # -------------------------------------------------------------------------
    # Save the result
    # -------------------------------------------------------------------------
    mesh_with_vorticity.save(output_file)
    print(f"Processed {input_file} -> {output_file}")

print("Computation completed successfully!")
