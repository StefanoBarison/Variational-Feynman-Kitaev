import numpy as np 
import json
import matplotlib.pyplot as plt 


from qiskit                 import IBMQ, Aer
from qiskit.utils           import QuantumInstance
from qiskit.opflow          import X, Y, Z, I


from vfk_functions    import *

# In this script we will measure observables over a state produced with VQE on the clock Hamiltonian

# As a first thing, load the result of VQE

data = json.load(open("data/VFK/VFK_depth2_2spin_2ancilla_dt1.0.dat"))

params = data["parameters"][-1]

print("Parameters: ", params)
print("We used "+str(len(params))+" parameters")
depth = data["ansatz_reps"][0]

energy = data["energies"][-1]
print("the configuration found has energy "+str(energy))

# Data of the system
n_spins   = data["spins"][0]
n_ancilla = data["ancilla_qubits"][0]
dt      = 1.0
times   = [i*dt for i in range(np.power(2,n_ancilla))]

print("Spins: ",n_spins)
print("Ancilla: ",n_ancilla)
# Observables data

obs = Pauli_op_single("Z",0,n_spins)
obs_vec = []


# Create a quantum instance

shots    = 1
backend  = Aer.get_backend('statevector_simulator')
instance = QuantumInstance(backend=backend,shots=shots)

for i in range(len(times)):
	res = measure_observable(obs,n_ancilla,i,example_ansatz,depth,params,instance)
	obs_vec.append(res)

obs_vec = np.asarray(obs_vec)

print("\nObservables")
print(obs_vec)

## Now plot the measured data and confront with real

exact    = json.load(open('data/Exact/OBS_exact_ising_'+str(n_spins)+'spin_J0.25_B1.dat'))
plt.plot(exact['times'][:61],exact['Sz'][:61],linestyle='dashed',color='black',label="Exact")
plt.plot(times,obs_vec[:,0],linestyle='',marker='o',label='Feynman-Kitaev VQE')
plt.xlabel(r"$t$")
plt.ylabel(r"$\langle \sigma^{z}_{1} \rangle$")
plt.legend()
plt.title(str(n_spins)+' spins and '+str(n_ancilla)+' ancillae')
plt.show()

