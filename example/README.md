# Eiffel Tower example

`lbm_tour_eiffel_bcs_null.msp`, `lbm_tour_eiffel_bcs_periodic.msp` and `lbm_tour_eiffel_bcs_rho.msp` simulate a fluid flow around a scaled model of the Eiffel Tower, imported as an R-shaped obstacle from `stl_files/toureiffel.stl` (`register_rshape`), with `wall_bounce_back` applied on its surface.

- **`lbm_tour_eiffel_bcs_null.msp`**: `0.6 m x 0.6 m x 2.0 m` domain on a `240 x 240 x 800` grid, non-periodic (`periodic: [false, false, false]`), tower placed at its base with a spherical perturbation of the distribution function (`set_distribution`, `quadrics: sphere`) added upstream to seed the flow. Neumann condition on the top/bottom planes, `lid_driven_cavity` (constant inflow velocity) on the four lateral planes. Runs for 5,000 iterations, output in `TourEiffelBCSNull/`.
- **`lbm_tour_eiffel_bcs_periodic.msp`**: same domain and seeding as above, periodic in the x/y directions (`periodic: [true, true, false]`), with only a Neumann condition on the top/bottom planes. Runs for 5,000 iterations, output in `TourEiffelPeriodic/`.
- **`lbm_tour_eiffel_bcs_rho.msp`**: elongated `5.0 m x 0.6 m x 2.0 m` domain on a `2000 x 240 x 800` grid to give the wake room to develop, periodic in y only (`periodic: [false, true, false]`). The flow is driven by a `rho` boundary condition imposing `delta_p: ±50 Pa` at the inlet/outlet planes (`plan_yz_0`, `plan_yz_l`) instead of a velocity condition, with Neumann still on the top/bottom planes; no upstream perturbation is needed since the pressure difference alone starts the flow. Runs for 50,000 iterations, ParaView output every 10,000 iterations.

## Run

```bash
./hippoLBM example/lbm_tour_eiffel_bcs_null.msp --omp_num_threads 4
./hippoLBM example/lbm_tour_eiffel_bcs_periodic.msp --omp_num_threads 4
./hippoLBM example/lbm_tour_eiffel_bcs_rho.msp --omp_num_threads 4
```

ParaView output is written every 100 iterations (`simulation_paraview_freq: 100`) for the null/periodic variants, and every 10,000 iterations for `lbm_tour_eiffel_bcs_rho.msp`, in the corresponding `output_directory`.

> **Note:** on an NVIDIA A100 GPU, writing the ParaView output files accounts for about 87% of the total run time. Increasing `simulation_paraview_freq` (or disabling ParaView output) is recommended to get a more representative measure of the LBM solver's performance.
>
> **Note:** `lbm_tour_eiffel_bcs_rho.msp` was benchmarked on 16x A100 GPUs, processing 386,275,041 LBM nodes over 50,000 timesteps in ~28.4 minutes.

# Pressurized sphere in water

`lbm_pressure_sphere_water.msp` simulates a spherical over-pressure released at the center of a fully periodic `2.0 m x 2.0 m x 2.0 m` box of water (`nuth: 1e-6 m2/s`, `avg_rho: 1000 kg/m3`), discretized on a `100 x 100 x 100` grid.

The fluid is initialized at rest (`set_distribution`, `value: 1.0`), then a `set_dp_pressure` operator adds a `delta_p: 1.0e6` Pa (1 MPa) over-pressure inside a sphere of radius `0.15 m` at the domain center, relative to the surrounding water (`delta_p: 0` leaves the fluid unperturbed). The resulting pressure pulse propagates as an acoustic wave through the domain and wraps around the periodic boundaries. The `mrt` collision operator is used for extra numerical stability, and the lattice celerity is set to `14800 m/s` (10x the physical speed of sound in water) to keep the induced density perturbation small.

## Run

```bash
./hippoLBM example/lbm_pressure_sphere_water.msp --omp_num_threads 4
```

ParaView output is written every 20 iterations in `PressureSphereWater/`; the `P` field is the pressure difference (Pa) relative to the water's reference state.

# Poiseuille flow

`lbm_poiseuille.msp` and `lbm_poiseuille_rho.msp` simulate the same plane Poiseuille flow: a `0.1 m x 0.1 m x 0.1 m` box discretized on a `30 x 30 x 30` grid (`nuth: 1e-3 m2/s`), with no-slip walls approximated by a Neumann condition on the top/bottom planes (`plan_xy_0`, `plan_xy_l`) and translational symmetry along y (`periodic: true`). The flow develops along x, and the resulting parabolic velocity profile is sampled along z at mid-channel (`plot_line_velocity`, checked by `plane_velocity_profile`).

The two variants differ only in how the flow is driven:

- **`lbm_poiseuille.msp`**: periodic in x (`periodic: [true, true, false]`), driven by a uniform body force `Fext: [9.512485e-05, 0, 0]` applied to the whole domain via `lbm_parameters`.
- **`lbm_poiseuille_rho.msp`**: open in x (`periodic: [false, true, false]`), driven instead by a `rho` boundary condition imposing a pressure difference at the inlet/outlet planes (`plan_yz_0`, `plan_yz_l`). For `celerity = 1` (the default, `dtLB = dx`), a uniform body force `Fext_x` over a channel of length `Lx` is exactly equivalent to a pressure drop `delta_p = Fext_x * Lx` split symmetrically around the reference state, giving `delta_p: [4.756243e-03, -4.756243e-03]` Pa here.

## Run

```bash
./hippoLBM example/lbm_poiseuille.msp --omp_num_threads 4
./hippoLBM example/lbm_poiseuille_rho.msp --omp_num_threads 4
```

Both run for 3,000 iterations, with ParaView output every 100 iterations in `PoiseuilleTestDir/` and `PoiseuilleRhoTestDir/` respectively.

> **Note on pressure units:** `rho`'s `delta_p` is parsed as an `onika::physics::Quantity`, whose unit table only knows base SI units (no `Pa`, `kPa`, `MPa`, `bar`, or `kbar` symbol). Express a pressure as the composite unit `kg/m/s^2` (1 Pa = 1 kg.m^-1.s^-2), scaling the numeric value for other pressure units:
>
> | Unit | Value in `kg/m/s^2` |
> |------|----------------------|
> | 1 Pa   | `1`     |
> | 1 kPa  | `1e3`   |
> | 1 MPa  | `1e6`   |
> | 1 bar  | `1e5`   |
> | 1 kbar | `1e8`   |
