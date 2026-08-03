# Eiffel Tower example

`lbm_tour_eiffel_bcs_null.msp` and `lbm_tour_eiffel_bcs_periodic.msp` simulate a fluid flow around a scaled model of the Eiffel Tower, imported as an R-shaped obstacle from `stl_files/toureiffel.stl` (`register_rshape`), with `wall_bounce_back` applied on its surface.

The domain is a `0.6 m x 0.6 m x 2.0 m` box discretized on a `240 x 240 x 800` grid, with the tower placed at its base and a spherical perturbation of the distribution function (`set_distribution` with `quadrics: sphere`) added upstream to seed the flow.

The two variants differ only in their lateral boundary conditions:

- **`lbm_tour_eiffel_bcs_null.msp`**: non-periodic domain (`periodic: [false, false, false]`), with a Neumann condition on the top/bottom planes and a `lid_driven_cavity` condition (constant inflow velocity) on the four lateral planes. Runs for 5,000 iterations, output in `TourEiffelBCSNull/`.
- **`lbm_tour_eiffel_bcs_periodic.msp`**: periodic domain in the x/y directions (`periodic: [true, true, false]`), with only a Neumann condition on the top/bottom planes. Runs for 5,000 iterations, output in `TourEiffelPeriodic/`.

## Run

```bash
./hippoLBM example/lbm_tour_eiffel_bcs_null.msp --omp_num_threads 4
./hippoLBM example/lbm_tour_eiffel_bcs_periodic.msp --omp_num_threads 4
```

ParaView output is written every 100 iterations (`simulation_paraview_freq: 100`) in the corresponding `output_directory`.

> **Note:** on an NVIDIA A100 GPU, writing the ParaView output files accounts for about 87% of the total run time. Increasing `simulation_paraview_freq` (or disabling ParaView output) is recommended to get a more representative measure of the LBM solver's performance.

# Pressurized sphere in water

`lbm_pressure_sphere_water.msp` simulates a spherical over-pressure released at the center of a fully periodic `2.0 m x 2.0 m x 2.0 m` box of water (`nuth: 1e-6 m2/s`, `avg_rho: 1000 kg/m3`), discretized on a `100 x 100 x 100` grid.

The fluid is initialized at rest (`set_distribution`, `value: 1.0`), then a `set_dp_pressure` operator adds a `delta_p: 1.0e6` Pa (1 MPa) over-pressure inside a sphere of radius `0.15 m` at the domain center, relative to the surrounding water (`delta_p: 0` leaves the fluid unperturbed). The resulting pressure pulse propagates as an acoustic wave through the domain and wraps around the periodic boundaries. The `mrt` collision operator is used for extra numerical stability, and the lattice celerity is set to `14800 m/s` (10x the physical speed of sound in water) to keep the induced density perturbation small.

## Run

```bash
./hippoLBM example/lbm_pressure_sphere_water.msp --omp_num_threads 4
```

ParaView output is written every 20 iterations in `PressureSphereWater/`; the `P` field is the pressure difference (Pa) relative to the water's reference state.
