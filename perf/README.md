# Perf benchmarks — 512x512x512

Sequential run commands for the different collision/streaming variants at the 512x512x512 grid
size. Copy-paste the block below (or run this file line by line); each `ccc_mprun` call blocks
until its job finishes, so they run one after another.

```sh
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512.msp --omp_num_threads 32

ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_inplace_bgk.msp --omp_num_threads 32
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_inplace_mrt.msp --omp_num_threads 32
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_pull_bgk.msp --omp_num_threads 32
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_pull_mrt.msp --omp_num_threads 32

ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_fuse_inplace_bgk.msp --omp_num_threads 32
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_fuse_inplace_mrt.msp --omp_num_threads 32
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_fuse_pull_bgk.msp --omp_num_threads 32
ccc_mprun -n 1 -c 32 -Q test -T 600 -m scratch,work -p a100 ./hippoLBM perf_couette_512x512x512_fuse_pull_mrt.msp --omp_num_threads 32
```
