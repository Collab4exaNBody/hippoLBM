# Vorticity Computation

This script computes the **vorticity field** from a velocity field stored in a parallel VTK (`.pvtr`) file using **PyVista**.

## Requirements

- Python >= 3.8
- PyVista

Install the required package with:

```bash
pip install pyvista
```

## Usage

Run the script by specifying the input and output directories:

```bash
python compute_vorticity.py \
    --input-dir /path/to/input \
    --output-dir /path/to/output
```

The script expects the input file to be named:

```
hippoLBM_0001047500.pvtr
```

The generated output file will be:

```
vorticity_output.pvtr
```

inside the specified output directory.

## Input

The input PVTR file must contain a vector field named:

```
U
```

This field is used to compute the vorticity.

## Output

The output file is a PVTR dataset identical to the input one, with an additional field containing the computed vorticity.

## Example

```bash
python compute_vorticity.py \
    --input-dir ./simulation \
    --output-dir ./results
```

## Notes

- The output directory is automatically created if it does not already exist.
- The computation relies on `pyvista.compute_derivative()` with the `vorticity=True` option.