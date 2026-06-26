---
title: 'HippoLBM: A HPC lightweight LBM software based on Onika'
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
    orcid: 0009-0009-5120-1308
    affiliation: 1    
affiliations:
 - name: CEA, DES, IRESNE, DEC, Cadarache F 13108 St-Paul-Lez-Durance
   index: 1
date: 26 June 2026
bibliography: paper.bib
---

# Summary

`HippoLBM` is ...

# Statement of need
<!--
`HippoLBM` is CFD code writen in C++ 20 ...
`HippoLBM` a pour objectif de proposer un outils performant sur CPU et GPU pour effectuer des couplages LBM+X en utilisant le formalisme Onika qui permet de créer des graphes d'exécution à partir d'une liste d'opérateur.

Un opérateur peut être l'appel à un kernel de calcul comme l'étape de collision BGK ou MRT, l'initialisation d'un champs, des sorties paraview ou n'importe quel étape ou liste d'étape lors du calcul. Dans `HippoLBM` nous cherchons à proposer une granularité fine de ces opérateurs pour pouvoir construire des couplages avec d'autres codes utilisant eux aussi le formalisme Onika. 

Le premier cas d'utilisation a été réalisé en couplant `HippoLBM` avec le code `exaDEM` pour effectuer des simulations DEM/LBM.

Concernant les fonctionnalités de performance, `HippoLBM` propose une parallélisation hybrid `MPI` + `X`, `X`=`OpenMP` ou `CUDA`, en utilisant les méthodes et stratégies classiques de parallélisation de la méthode LBM (décomposition spatial du domaine, optimisation GPU TODO). Néanmoins, certaines stratégies comme l'utilisation de méthod raffinement adaptatifs de maillage ou la fusion automatique de kernel n'ont pas été intégrées.
-->

`HippoLBM` is a CFD code written in C++20 using the Lattice Boltzmann Method (LBM) that aims to provide a high-performance tool on both CPU and GPU for LBM+X coupling, using the onika formalism, which enables the construction of execution graphs from a list of operators.
An operator can be a call to a compute kernel such as the BGK or MRT collision step, a field initialization, a ParaView output, or any step or sequence of steps within the computation. In `HippoLBM`, we seek to provide fine-grained operators in order to build couplings with other codes that also use the `Onika` formalism. The first use case was achieved by coupling HippoLBM with the exaDEM code to perform DEM/LBM simulations.

Regarding performance features, `HippoLBM` provides hybrid MPI+X parallelization, where X is either OpenMP or CUDA, using standard LBM parallelization methods and strategies (spatial domain decomposition, GPU optimization TODO). However, certain strategies such as adaptive mesh refinement or automatic kernel fusion have not yet been integrated.

# State of the field                                                                                                                  

`ProLB`, ...

# Software design
<!--
`HippoLBM`'s design philosophy is to decompose the LBM simulations on a list of `onika` operators. 
`HippoLBM` est composé en plusieurs plugins, actuellement tout les plugins présents forment le coeurs d'`HippoLBM`:
plugin grid: Ce plugin contient la plupart des structures de données comme les champs, les données sur le domaine, les paramètres LBM et propose tous les operateurs  permettant de modififier/initialiser ces structures de données, notamment l'équilibrage de charge (block).
plugin collision: Ce plugin permet d'appliquer les étapes élémentaires de la LBM comme l'application de l'opérateur de collision BGK ou MRT, la phase de streaming ou le calul des quantités macros comme la vitesse et la perssion.
plugin bcs: Ce plugin contient les noyaux de calculs pour appliquer les conditions limites comme des conditions de neumann utilisé pour des cas tests académique comme un écoulement de Couette ou de Poiseuille, bounce back pour modéliser des solides, ou des conditions limites spécifiques pour mettre en place des cavités entraînées.
plugin IO: Ce plugin est actuellement utlisé pour afficher des logs et effectuer des sorties paraview (post traitement). Il a aussi vocation à évoluer pour intégrer des analyses in-situ.
plugin Prepo: Ce plugin propose de pré-initialiser les champs pour des régimes très précis comme par exemple un double couette.
plugin Obstacle: Ce plugin permet de placer des objects solides inamovible comme des murs.
-->



`HippoLBM`'s design philosophy is to decompose LBM simulations into a list of `Onika` operators. To that end, it is organized into several plugins, all of which currently form the core of HippoLBM:
- `grid`: This plugin contains most of the data structures, such as fields, domain data, and LBM parameters, and provides all operators for modifying and initializing these data structures, including load balancing (block).
- `collision`: This plugin applies the elementary steps of LBM, such as the BGK or MRT collision operator, the streaming phase, and the computation of macroscopic quantities (e.g., velocity and pressure).
- `bcs`: This plugin contains the compute kernels for applying boundary conditions (e.g., Neumann conditions for Couette or Poiseuille flows, bounce-back for solid boundaries, or lid-driven cavity setups).
- `io`: This plugin is currently used to display logs and produce ParaView output files for post-processing. Future developments will extend it to support in-situ analysis.
- `prepo`: This plugin provides pre-initialization of fields for specific flow regimes, such as double Couette flow.
- `obstacle`: This plugin allows placing fixed solid objects, such as walls, within the simulation domain.

# Research impact statement

Talk about couplings ...

# AI usage disclosure

No generative AI tools were used in the design and development of this software, but they have been used for refactoring and renaming classes.
Generative AI tools were used to generate Doxygen code and to translate texts for website documentation.

# Acknowledgements

This work was performed using HPC resources from CCRT funded by the CEA/DEs simulation program. `HippoLBM` is part of the `PLEIADES` platform which has been developped in collaboration with the French nuclear industry - mainly CEA, EDF, and Framatome - for simulation of fuel elements.

# References
