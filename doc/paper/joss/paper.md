---
title: 'HippoLBM: Lightweight HPC LBM software based on Onikaa'
tags:
  - Lattice Boltzmann
  - C++
  - CUDA
  - MPI
  - Coupling
authors:
  - name: Raphaël Prat
    orcid: 0009-0002-3808-5401
    affiliation: 1
  - name: Lhassan Amarsid
    orcid: 0009-0009-5120-1308
    affiliation: 1
  - name: Vincent Topin
    orcid: 0009-0009-5897-4979
    affiliation: 1    
  - name: Guillaume Bareigts
    orcid: 0000-0002-9444-9858
    affiliation: 1       
  - name: Bruno Collard
    orcid: 0009-0009-2152-3816 
    affiliation: 1     
affiliations:
 - name: CEA, DES, IRESNE, DEC, Cadarache F 13108 St-Paul-Lez-Durance
   index: 1
date: 26 June 2026
bibliography: paper.bib
---

# Summary

`HippoLBM` is a high-performance software for simulating fluid flows with the Lattice Boltzmann Method (LBM), targeting both multicore CPUs and GPUs. Built on the `Onika` framework, it decomposes a simulation into a sequence of independent operators - collision, streaming, or boundary conditions - that can be freely assembled into custom simulation workflows. This operator-based design makes `HippoLBM` well suited for coupling LBM with other numerical methods, such as the Discrete Element Method (DEM), to simulate fluid-particle systems relevant to nuclear engineering and other research or industrial applications. `HippoLBM` is intended to run large-scale, three-dimensional LBM simulations, either standalone or coupled with other physics, on modern hybrid MPI+CPU/GPU supercomputers.

# Introduction

The Lattice Boltzmann Method (LBM) [@PhysRevLett.61.2332] is a numerical method for computational fluid dynamics (CFD) based on a mesoscopic description of fluid dynamics. Unlike classical methods for solving the Navier-Stokes equations, which directly describe the evolution of macroscopic quantities such as velocity and pressure, LBM governs the spatio-temporal evolution of distribution functions representing the statistical behavior of the particles making up the fluid. This approach originates from the kinetic theory of gases [@chapman1916mathematical]. Instead of directly solving the macroscopic equations of fluid motion, LBM describes the evolution of distribution functions associated with fictitious particles moving along a discrete set of directions defined on a regular lattice.

One of the main advantages of LBM lies in its local and explicit formulation. Computations are performed only between neighboring lattice nodes, which enables efficient parallelization on modern computing architectures such as multicore CPUs and GPU accelerators. This feature makes the method particularly well suited to high-resolution three-dimensional simulations, and it inherently exposes fine-grained parallelism since updates at each lattice node are locally independent. Its use of a regular Cartesian mesh further facilitates efficient GPU implementations, enabling simulations at a very large scale, ranging up to billions of lattice nodes.

LBM also offers great flexibility for representing complex geometries and boundary conditions that are difficult to handle with classical methods. Various collision models have been developed to improve numerical stability and extend the range of applications of the method, including Multiple Relaxation Time (MRT) models [@PhysRevE.61.6546], entropic models, and cumulant-based approaches [@krueger2017lattice]. Despite its many advantages, LBM has some limitations. Its classical formalism relies on a low-compressibility assumption and is therefore mainly suited to flows characterized by low Mach numbers. In addition, difficulties can arise in situations with strong pressure gradients or highly turbulent regimes. These limitations are nonetheless the subject of extensive ongoing work aimed at improving numerical stability and broadening the range of application of the method.

This method remains particularly attractive for multiphysics simulations, as it can be easily coupled with other physical models. In particular, its combination with discrete particle methods through the Immersed Boundary Method (IBM) provides an efficient framework for simulating complex fluid-particle interactions relevant for a wide range of industrial or research applications. To implement such couplings, we developed a framework that expresses each elementary operation (I/O, numerical schemes, analyses) as an operator and connects operators via slots. In this paper, we concentrate on the `HippoLBM` code, derived from legacy LBM-DEM software and refactored for GPU execution and hybrid MPI+GPU parallelization.

# Principle of the LBM

The LBM discretizes the Boltzmann equation on a regular lattice, replacing the continuous distribution function with a finite set of populations $f_i(\mathbf{r},t)$ associated with discrete velocity directions $\mathbf{e}_i$ [@krueger2017lattice]. In the Bhatnagar-Gross-Krook (BGK) model used by `HippoLBM`, the collision and streaming steps read:

$$
f_i^*(\mathbf{r},t) = f_i(\mathbf{r},t) - \frac{\Delta t}{\tau}\left(f_i(\mathbf{r},t)-f_i^{eq}(\mathbf{r},t)\right),
\qquad
f_i(\mathbf{r}+\mathbf{e}_i\Delta t,t+\Delta t) = f_i^*(\mathbf{r},t),
$$

where $\tau$ is the relaxation time and $f_i^{eq}$ is the equilibrium distribution, obtained from a second-order expansion of the Maxwell-Boltzmann distribution and valid in the low Mach number limit ($Ma \ll 1$) [@krueger2017lattice]. The macroscopic density $\rho$ and momentum $\rho\mathbf{u}$ are recovered as the zeroth- and first-order moments of $f_i$, and the fluid's kinematic viscosity is directly related to $\tau$. `HippoLBM` also implements the Multiple Relaxation Time (MRT) model, which relaxes the collision operator's moments independently rather than using a single relaxation time $\tau$, improving numerical stability over BGK at higher Reynolds numbers [@PhysRevE.61.6546]. For both models, a Chapman-Enskog asymptotic expansion shows that the scheme recovers the macroscopic mass and momentum conservation equations in the low Mach number limit [@chapman1916mathematical; @krueger2017lattice].

# Statement of need


`HippoLBM` is a C++20 LBM code that aims to provide a high-performance tool for coupling LBM with other numerical methods on both CPU and GPU, using the `Onika` formalism [@carrard2023exanbody] to build execution graphs from a list of operators.
In `HippoLBM`, an operator can be a compute kernel call such as the BGK or MRT collision step, a field initialization, a ParaView output, or any other step or sequence of steps in the computation. We target fine operator granularity to enable couplings with other codes that also use the `Onika` formalism. The first use case was coupling `HippoLBM` with the `exaDEM` code [@prat2025exadem] for DEM-LBM simulations using R-shaped particles.

![a) Lid driven cavity simulation. b) Example using obstacles defined by quadrics. c) Von Kármán vortex street simulation. \label{fig:examples}](./groupir.png){width=70%} 

Regarding performance, `HippoLBM` supports hybrid MPI+X parallelization, where X is either OpenMP or CUDA, and relies on standard LBM parallelization strategies (spatial domain decomposition, GPU optimization [@tran2017performance]). However, some strategies such as adaptive mesh refinement or automatic kernel fusion [@mahmoud2024optimized] are not yet implemented. `HippoLBM` has been tested on 192 NVIDIA A100 GPUs and can handle around 69 billion LB points and proposes near-perfect scaling up to 128 nodes, see \autoref{fig:perf}.


![Number of Million Lattice Updates per Second (MLUPS) in strong scaling for different domain sizes of a Couette Flow simulation. This benchmark was conducted on NVIDIA A100 GPUs with CUDA 12.4 on the CCRT Topaze supercomputer. \label{fig:perf}](./perf.png){width=60%} 



# State of the field                                                                                                                  


In the field of codes using the 3D Lattice Boltzmann Method, several codes offer more advanced physical capabilities than `HippoLBM`, such as open source codes `OpenLB` [@heuveline2007openlb], which provides a broad, general-purpose set of physical models (e.g., thermal, particulate, and free-surface flows), or `LBMSaclay` [@cartalade2016lattice], which enables multiphase simulations. Non-open-source codes such as `ProLB` [@feng2021prolb], which can simulate compressible fluids, or `PowerFLOW`®.


`HippoLBM` differs from the state of the art mainly in its design rather than in its physical or HPC capabilities, which can be further enriched in the future in order to integrate into complex, multi-physics ecosystems. Note that waLBerla [@bauer2021walberla] and the Palabos-LIGGGHTS coupling [@latt2021palabos] offer similar multiphysics capabilities with HPC features.


# Software design


`HippoLBM`'s design philosophy is to decompose LBM simulations into a list of `Onika` operators. To that end, it is organized into several plugins, all of which currently form the core of HippoLBM:

- `grid`: This plugin contains most of the data structures, such as fields, domain data, and LBM parameters, and provides all operators for modifying and initializing these data structures, including load balancing (block).
- `collision`: This plugin applies the elementary steps of LBM, such as the BGK or MRT collision operator, the streaming phase, and the computation of macroscopic quantities (e.g., velocity and pressure).
- `bcs`: This plugin contains the compute kernels for applying boundary conditions (e.g., Neumann conditions for Couette or Poiseuille flows, bounce-back for solid boundaries, or lid-driven cavity setups).
- `io`: This plugin is currently used to display logs and produce ParaView output files for post-processing. Future developments will extend it to support in-situ analysis.
- `prepro`: This plugin provides pre-initialization of fields for specific flow regimes, such as double Couette flow.
- `obstacle`: This plugin allows placing fixed solid objects, such as walls, rounded-shape particles (spheropolyhedra), or geometries defined by quadrics, see \autoref{fig:examples}.b, within the simulation domain. 

![a) Overview of the HippoLBM plugins built on top of the Onika runtime. b) Example of an operator sequence, colored by plugin, executed within the time loop. \label{fig:design}](./LBMDesign.pdf){width=100%}

# Research impact statement

`HippoLBM` originates from the extraction of the LBM component out of a legacy, monolithic LBM-DEM code used to study granular flows immersed in a viscous fluid [@amarsid2017viscoinertial]. This extraction served two purposes: preparing a dedicated HPC (MPI+GPU) port of the LBM solver, and making these developments reusable beyond their original DEM-coupling context. Because `HippoLBM` and `exaDEM` [@prat2025exadem] are both built on the Onika framework, they can be coupled with minimal effort, as illustrated by the DEM-LBM simulations of R-shaped particles mentioned in the Statement of need. Through this same Onika interface, HippoLBM can also be coupled to other physical models, such as the Material Point Method (MPM) [@xiao2026volume], the Finite Element Method (FEM), or the Finite Difference Method (FDM), targeting the broader community working on resolved fluid-particle systems.

# AI usage disclosure

Generative AI tools were not used in the algorithmic design of this software. They were used for peripheral development tasks: refactoring and renaming classes, generating post-processing Python scripts and Doxygen documentation, and translating text for the website documentation.

# Acknowledgements

This work was performed using HPC resources from CCRT funded by the CEA/DES simulation program. `HippoLBM` is part of the `PLEIADES` platform which is developed in collaboration with the French nuclear industry - mainly CEA, EDF, and Framatome - for the simulation of nuclear fuel behavior.

# References
