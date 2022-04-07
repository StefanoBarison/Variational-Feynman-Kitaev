import numpy as np
import matplotlib.pyplot as plt
import json
import math


from qiskit                       import QuantumCircuit, IBMQ, Aer
from qiskit.circuit               import ParameterVector
from qiskit.providers.aer         import QasmSimulator
from qiskit.utils                 import QuantumInstance
from qiskit.opflow                import X, Y, Z, I


if __name__ == "__main__":
	from vfk_functions        import *


	## This is the main program for history-state variational determination


	## System specification

	n_spins    = 2
	n_ancilla  = 2
	n_qubits   = n_spins + n_ancilla
	J        = 0.25
	B        = 1
	dt       = 1.0
	steps    = 2

	
	## Create the Hamiltonian

	H_op  = create_clock_hamiltonian(n_spins,n_ancilla,dt,J,B)
	
	# Ansatz characteristics for the system
	depth    = 2

	initial_point = np.ones(depth*2*int(n_ancilla*(n_spins**2 +n_spins)))   


	print("The system Hamiltonian is")
	print(H_op)


	# Initialize a backend for the calculation

	shots    = 1
	backend  = Aer.get_backend('statevector_simulator')
	instance = QuantumInstance(backend=backend,shots=shots)

	opt      = 'adam' # 'sgd'
	grad     = 'param_shift' #'param_shift' 'spsa'

	
	VFK(n_qubits=n_qubits,
	    ancilla_qubits=n_ancilla,
	    operator=H_op,
	    ansatz= example_ansatz,
	    ansatz_reps=depth,
		init_params=initial_point,
		instance=instance,
		max_iter=1000,
		opt=opt,
		grad=grad,
		filename ="data/VFK/VFK_depth"+str(depth)+"_"+str(n_spins)+"spins_"+str(n_ancilla)+"ancilla_dt"+str(dt)+".dat"
		)

