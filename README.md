# Variational-Feynman-Kitaev
This repository contains the source code for the VFK algorithm introduced in _ArXiv:2204._


The method has been implemented both in Python and Julia, using different libraries for quantum computing.
We strongly suggest to use Julia implementations.

Every directory contains methods to construct the Feynman-Kitaev Hamiltonian for the Transverse Field Ising Model and to find the ground state variationally using the VQE.

---

## Python - Qiskit implementation

We implemented the method in Python using the [Qiskit framework](https://qiskit.org).

At the moment, the Qiskit version required is


|qiskit                   | 0.32.1  |
|-------------------------|---------|
|qiskit-aer               | 0.9.1   |
|qiskit-aqua              | 0.9.5   |
|qiskit-ignis             | 0.6.0   |
|qiskit-ibmq-provider     | 0.18.1  |
|qiskit-ignis             | 0.6.0   |
|qiskit-nature            | 0.2.2   |
|qiskit-terra             | 0.18.3  |


---

## Julia - Yao.jl implementation

We implented the method in Julia using the [Yao.jl](https://yaoquantum.org/) package.

The corresponding directory contains a `Project.toml` and a `Manifest.toml` file in order to reproduce the Julia project.

---

## Julia - PastaQ.jl implementation - Available Soon!

We will include an implentation of the VFK method in Julia using the [PastaQ.jl](https://github.com/GTorlai/PastaQ.jl) package.