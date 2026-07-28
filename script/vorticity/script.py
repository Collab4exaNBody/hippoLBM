import argparse
from pathlib import Path

import pyvista as pv


# -----------------------------------------------------------------------------
# Command-line arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compute the vorticity field from a PVTR file.")
parser.add_argument(
    "--input-dir",
    type=Path,
    required=True,
    help="Directory containing the input PVTR file.",
)
parser.add_argument(
    "--output-dir",
    type=Path,
    required=True,
    help="Directory where the output PVTR file will be saved.",
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Input and output files
# -----------------------------------------------------------------------------
input_file = args.input_dir / "hippoLBM_0001047500.pvtr"
output_file = args.output_dir / "vorticity_output.pvtr"

# -----------------------------------------------------------------------------
# Load the master PVTR file
# -----------------------------------------------------------------------------
mesh = pv.read(input_file)

# -----------------------------------------------------------------------------
# Compute the vorticity from the velocity field "U"
# -----------------------------------------------------------------------------
mesh_with_vorticity = mesh.compute_derivative(
    scalars="U",
    vorticity=True,
)

# -----------------------------------------------------------------------------
# Save the result
# -----------------------------------------------------------------------------
args.output_dir.mkdir(parents=True, exist_ok=True)
mesh_with_vorticity.save(output_file)

print("Computation completed successfully!")
print(f"Output written to: {output_file}")
