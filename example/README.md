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
