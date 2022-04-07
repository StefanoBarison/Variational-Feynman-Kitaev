# Variational Feynman-Kitaev using Qiskit

This directory is organized as follows:

- the `main.py` file contains the script to run the VFK algorithm on the Transverse Field Ising Model, provided a number of spins and of clock qubits
- the `vfk_functions.py` file contains custom functions to create the Feynman-Kitaev Hamiltonian, the ansatz circuit and measure observables
- the `measure_observables.py` file contain a script to measure $\sigma^1_z$ on the variational history state and plot it using matplotlib