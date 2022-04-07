# Variational Feynman-Kitaev using Yao.jl

When in the directory, use `julia --project=.` to open a Julia session and `]instatiate` before running the code.

This directory is organized as follows:

- the `main.jl` file contains the script to run the VFK algorithm on the Transverse Field Ising Model, provided a number of spins and of clock qubits
- the `vfk_functions.jl` file contains custom functions to create the Feynman-Kitaev Hamiltonian, the ansatz circuit and measure observables
- the `measure_observables.jl` file contain a script to measure $\sigma^1_z$ on the variational history state and plot it using PyPlot.jl