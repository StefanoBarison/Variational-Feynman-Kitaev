## in this script we will try to implement clock hamiltonian measurements on quantum hardware

import numpy as np 
import json
import math
import functools
import itertools
import matplotlib.pyplot as plt 
from scipy   import  linalg as LA 


from qiskit                    import QuantumCircuit, Aer, ClassicalRegister, QuantumRegister
from qiskit.circuit            import ParameterVector
from qiskit.opflow             import StateFn, AerPauliExpectation, PauliExpectation, CircuitOp, MatrixOp, CircuitSampler
from qiskit.opflow             import X, Y, Z, I
from qiskit.opflow             import PauliOp, SummedOp, PauliSumOp
from qiskit.utils              import QuantumInstance
from qiskit.quantum_info       import Pauli, PauliTable, SparsePauliOp
from qiskit.quantum_info       import DensityMatrix


## Useful function to create gray code of dimension n 

def gray_code(n):
    def gray_code_recurse (g,n):
        k=len(g)
        if n<=0:
            return

        else:
            for i in range (k-1,-1,-1):
                char='1'+g[i]
                g.append(char)
            for i in range (k-1,-1,-1):
                g[i]='0'+g[i]

            gray_code_recurse (g,n-1)

    g=['0','1']
    gray_code_recurse(g,n-1)
    return g

## Useful function to create binary code of dimension n 

def binary_code(n):
	# This function gives all the binary string for numbers up to n bits
	code = []
	for j in range(2**n):
		code.append(format(j, '0'+str(n)+'b'))

	return code

#############################################

def custom_gate(params):
	g_qr = QuantumRegister(2)
	g_qc = QuantumCircuit(g_qr, name='G')

	g_qc.rx(params[0],g_qr[0])
	g_qc.ry(params[1],g_qr[0])
	g_qc.rx(params[2],g_qr[1])
	g_qc.ry(params[3],g_qr[1])
	g_qc.cx(g_qr[0],g_qr[1])

	return g_qc


# The ansatz presented in the article

def example_ansatz(n_qubits,ancilla,reps,p):

    n_spins = n_qubits - ancilla
    count = 0
    
    circ_qr  = QuantumRegister(n_qubits)
    circ     = QuantumCircuit(circ_qr)
    ent_list = [(l,m) for l in range(ancilla,n_qubits) for m in range(l+1,n_qubits)]

    circ.h([i for i in range(ancilla)])
    
    for d in range(reps):
        for i in range(ancilla):
            for j in range(ancilla,n_qubits):
                circ.append(custom_gate([p[count + k] for k in range(4)]).to_instruction(),[circ_qr[i],circ_qr[j]])
                count = count +4
            circ.barrier()
            for (l,m) in ent_list:
                circ.append(custom_gate([p[count + k] for k in range(4)]).to_instruction(),[circ_qr[l],circ_qr[m]])
                count = count +4
            circ.barrier()
            

    return circ

## Coordinate basis vector

def ei(i,n):
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]


#############################################
#Functions to create Paulis

def Pauli_op_single(pauli, i,n_qubits):
	from qiskit.opflow            import Z, I, X, Y
	p_list = [I for j in range(n_qubits)]
	if pauli == 'X':
		p_list[i] = X		
	if pauli == 'Y':
		p_list[i] = Y	
	if pauli == 'Z':
		p_list[i] = Z

	p = p_list[0]

	for a in range(1,len(p_list)):
		p = p^p_list[a]

	return p

def Pauli_op_double(pauli1,pauli2, i1,i2,n_qubits):
	p1 = Pauli_op_single(pauli1, i1,n_qubits)
	p2 = Pauli_op_single(pauli2, i2,n_qubits)

	p = p1.compose(p2)

	return p

def generate_pauli(idx_x,idx_z,n):
	'''
	Args:
		n (integer)
		idx (list)
	Returns:
		tensor product of Pauli operators acting on qubits in idx
	'''

	xmask = [0]*n
	zmask = [0]*n
	for i in idx_x : xmask[i] = 1
	for i in idx_z : zmask[i] = 1

	a_x = np.asarray(xmask,dtype =np.bool)
	a_z = np.asarray(zmask,dtype =np.bool)

	return Pauli((a_z,a_x))

#############################################

## Projectors

def projector_zero(n_qubits):
	from qiskit.opflow            import Z, I

	prj_list = [0.5*(I+Z) for i in range(n_qubits)]
	prj = prj_list[0]

	for a in range(1,len(prj_list)):
		prj = prj^prj_list[a]

	
	
	return prj

def projector_clock(n_qubits,time):
	from qiskit.opflow            import Z, I

	prj_list = [I for i in range(n_qubits)]

	gray_string = gray_code(n_qubits)[time]

	for i in range(len(gray_string)):


		if gray_string[i] == '0':
			prj_list[i] = 0.5*(I+Z)
		if gray_string[i] == '1':
			prj_list[i] = 0.5*(I-Z)
	

	prj = prj_list[0]

	for a in range(1,len(prj_list)):
		prj = prj^prj_list[a]

	return prj


# Projectors with binary code

def projector_clock_binary(n_qubits,time):
	from qiskit.opflow            import Z, I

	prj_list = [I for i in range(n_qubits)]

	gray_string = format(time, '0'+str(n_qubits)+'b')

	for i in range(len(gray_string)):


		if gray_string[i] == '0':
			prj_list[i] = 0.5*(I+Z)
		if gray_string[i] == '1':
			prj_list[i] = 0.5*(I-Z)
	

	prj = prj_list[0]

	for a in range(1,len(prj_list)):
		prj = prj^prj_list[a]

	return prj

#############################################


# Create the Hamiltonian


def create_clock_hamiltonian(spin_qubits,ancilla_qubits,dt,J,B):
    # This function creates the qubit hamiltonian of Feynman-Kitaev directly as pauli op
    # Initial projector term
    C0 = ((Pauli_op_single('I',0,spin_qubits)-1*projector_zero(spin_qubits)).reduce() ^ projector_zero(ancilla_qubits)).reduce()
    
    # Local projectors terms
    C1 = 0.5*((Pauli_op_single('I',0,spin_qubits)^projector_clock(ancilla_qubits,0))+ (Pauli_op_single('I',0,spin_qubits)^projector_clock(ancilla_qubits,2**ancilla_qubits-1))).reduce()
    for i in range(1,2**ancilla_qubits-1):
        C1 += (Pauli_op_single('I',0,spin_qubits)^projector_clock(ancilla_qubits,i))
        C1.reduce()
    # Now the evolution part:
    # 1) Trotter term 
    #Udt = Pauli_op_single('I',0,spin_qubits).reduce()
    Udt = (np.cos(J*dt)*Pauli_op_single('I',0,spin_qubits) -1j*np.sin(J*dt)*Pauli_op_double('Z','Z',0,1,spin_qubits)).reduce()
    for i in range(1,spin_qubits-1):
        Udt = Udt.compose((np.cos(J*dt)*Pauli_op_single('I',0,spin_qubits) -1j*np.sin(J*dt)*Pauli_op_double('Z','Z',i,i+1,spin_qubits)).reduce())
    Udt.reduce()
    for i in range(spin_qubits):
        t = np.cos(B*dt)*Pauli_op_single('I',0,spin_qubits)-1j*np.sin(B*dt)*Pauli_op_single('X',i,spin_qubits)
        Udt = Udt.compose(t)
    Udt.reduce()
    
    Udt_dag = Udt.adjoint()
    
    # 2) complete evolution
    
    C2 = 0
    # We must find the qubit that is different 
    gray_string = gray_code(ancilla_qubits)
    
    ## Pay attention to the order
    for j in range(len(gray_string)-1):
        g1 = gray_string[j]
        g2 = gray_string[j+1]

        # Construct the projector on g1
        exc_list = [I for i in range(ancilla_qubits)]
        for i in range(len(g1)):

            if g1[i] == '0':
                exc_list[i] = 0.5*(I+Z)
            if g1[i] == '1':
                exc_list[i] = 0.5*(I-Z)
        
        # Find out which bit changes
        exc_site = 0
        for g in range(len(g1)):
            if g1[g] != g2[g]:
                exc_site = g #len(g1)-(g+1) #to reverse
                
        
        if int(g2[exc_site])>int(g1[exc_site]):
            exc_list[exc_site] = 0.5*(X+1j*Y)
        else:
            exc_list[exc_site] = 0.5*(X-1j*Y)
            
        exc_op = exc_list[0]

        for a in range(1,len(exc_list)):
            exc_op = exc_op^(exc_list[a])
        
        exc_op = exc_op.reduce()  # This is |t><t+1|
        exc_op_dag = exc_op.adjoint() # this is |t+1><t|
        
        
        ## Now we can create the whole operator
        
        C2 += ((Udt)^(exc_op_dag)).reduce() + ((Udt_dag)^(exc_op)).reduce()
        C2 = C2.reduce()
        
    # Return the final operator
    C = (C0 + C1 -0.5*C2).reduce()
    return C


def create_clock_hamiltonian_binary(spin_qubits,ancilla_qubits,dt,J,B):
    # This function creates the qubit hamiltonian of Feynman-Kitaev directly as pauli op
    # Initial projector term
    C0 = ((Pauli_op_single('I',0,spin_qubits)-1*projector_zero(spin_qubits)).reduce() ^ projector_zero(ancilla_qubits)).reduce()
    
    # Local projectors terms
    C1 = 0.5*((Pauli_op_single('I',0,spin_qubits)^projector_clock_binary(ancilla_qubits,0))+ (Pauli_op_single('I',0,spin_qubits)^projector_clock_binary(ancilla_qubits,2**ancilla_qubits-1))).reduce()
    for i in range(1,2**ancilla_qubits-1):
        C1 += (Pauli_op_single('I',0,spin_qubits)^projector_clock_binary(ancilla_qubits,i))
        C1.reduce()
    # Now the evolution part:
    # 1) Trotter term 
    #Udt = Pauli_op_single('I',0,spin_qubits).reduce()
    Udt = (np.cos(J*dt)*Pauli_op_single('I',0,spin_qubits) -1j*np.sin(J*dt)*Pauli_op_double('Z','Z',0,1,spin_qubits)).reduce()
    for i in range(1,spin_qubits-1):
        Udt = Udt.compose((np.cos(J*dt)*Pauli_op_single('I',0,spin_qubits) -1j*np.sin(J*dt)*Pauli_op_double('Z','Z',i,i+1,spin_qubits)).reduce())
    Udt.reduce()
    for i in range(spin_qubits):
        t = np.cos(B*dt)*Pauli_op_single('I',0,spin_qubits)-1j*np.sin(B*dt)*Pauli_op_single('X',i,spin_qubits)
        Udt = Udt.compose(t)
    Udt.reduce()
    
    Udt_dag = Udt.adjoint()
    
    # 2) complete evolution
    
    C2 = 0
    # We must find the qubit that is different 
    bin_string = binary_code(ancilla_qubits)
    
    ## Pay attention to the order
    for j in range(len(bin_string)-1):
        g1 = bin_string[j]
        g2 = bin_string[j+1]

        # Construct the projector on g1
        exc_list = [I for i in range(ancilla_qubits)]

        for k in range(len(g1)):
            if int(g1[k]) - int(g2[k]) == -1:
                exc_list[k] = 0.5*(X+1j*Y)
            elif int(g1[k]) - int(g2[k]) == +1:
            	exc_list[k] = 0.5*(X-1j*Y)
            elif int(g1[k]) - int(g2[k]) == 0 & int(g1[k]) == 0:
            	exc_list[k] = 0.5*(I+Z)
            elif int(g1[k]) - int(g2[k]) == 0 & int(g1[k]) == 1:
            	exc_list[k] = 0.5*(I-Z)
			
            
        exc_op = exc_list[0]

        for a in range(1,len(exc_list)):
            exc_op = exc_op^(exc_list[a])
        
        exc_op = exc_op.reduce()  # This is |t><t+1|
        exc_op_dag = exc_op.adjoint() # this is |t+1><t|
        
        
        ## Now we can create the whole operator
        
        C2 += ((Udt)^(exc_op_dag)).reduce() + ((Udt_dag)^(exc_op)).reduce()
        C2 = C2.reduce()
        
    # Return the final operator
    C = (C0 + C1 -0.5*C2).reduce()
    return C




#######################################################
#Gradient functions

## Parameter shift 

def energy_and_gradient(state_wfn,parameters,parameter_vec,expectation,sampler):
	## This function evaluates the energy and the gradient during a VQE calculation

	nparameters = len(parameters)


	E    = np.zeros(2)
	grad = np.zeros((nparameters,2))

	# First create the dictionary for overlap
	values_dict = [dict(zip(parameter_vec[:], parameters.tolist()))]
		

	# Then the values for the gradient
	for i in range(nparameters):
		values_dict.append(dict(zip(parameter_vec[:] , (parameters + ei(i,nparameters)*np.pi/2.0).tolist())))
		values_dict.append(dict(zip(parameter_vec[:] , (parameters - ei(i,nparameters)*np.pi/2.0).tolist())))


	results = []
	for values in values_dict:
		sampled_op = sampler.convert(state_wfn,params=values)

		mean  = sampled_op.eval().real
		est_err = 0


		if ( not sampler.quantum_instance.is_statevector):
			shots    = sampler.quantum_instance.run_config.shots
			variance = expectation.compute_variance(sampled_op).real
			est_err  = np.sqrt(variance/shots)

		results.append([mean,est_err])

	E = np.zeros(2)
	g = np.zeros((nparameters,2))

	E[0],E[1] = results[0]

	for i in range(nparameters):
		rplus  = results[1+2*i]
		rminus = results[2+2*i]
		# G      = (Ep - Em)/2
		# var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
		g[i,:] = (rplus[0]-rminus[0])/2.0,np.sqrt(rplus[1]**2+rminus[1]**2)/2.0

	return E, g


## SPSA 

def energy_and_gradient_spsa(state_wfn,parameters,parameter_vec,expectation,sampler,count):
	## This function evaluates the energy and the gradient during a VQE calculation

	nparameters = len(parameters)
	E    = np.zeros(2)
	grad = np.zeros((nparameters,2))

	# Define hyperparameters
	c  = 0.1
	a  = 0.16
	A  = 1
	alpha  = 0.602
	gamma  = 0.101

	a_k = a/np.power(A+count,alpha)
	c_k = c/np.power(count,gamma)

	# Determine the random shift

	delta = np.random.binomial(1,0.5,size=nparameters)
	delta = np.where(delta==0, -1, delta) 
	delta = c_k*delta

	#delta = np.random.normal(loc=0,scale=c,size=nparameters)

	# First create the dictionary for overlap
	values_dict = [dict(zip(parameter_vec[:], parameters.tolist()))]
		

	# Then the values for the gradient approximation
	
	values_dict.append(dict(zip(parameter_vec[:] , (parameters + delta).tolist())))
	values_dict.append(dict(zip(parameter_vec[:] , (parameters - delta).tolist())))

	results = []
	for values in values_dict:
		sampled_op = sampler.convert(state_wfn,params=values)

		mean  = sampled_op.eval().real
		est_err = 0


		if ( not sampler.quantum_instance.is_statevector):
			shots    = sampler.quantum_instance.run_config.shots
			variance = expectation.compute_variance(sampled_op).real
			est_err  = np.sqrt(variance/shots)

		results.append([mean,est_err])

	E = np.zeros(2)
	g = np.zeros((nparameters,2))

	E[0],E[1] = results[0]

	# and the gradient
	rplus  = results[1]
	rminus = results[2]
	
	for i in range(nparameters):
		# G      = (Ep - Em)/2Δ_i
		# var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
		g[i,:] = a_k*(rplus[0]-rminus[0])/(2.0*delta[i]),np.sqrt(rplus[1]**2+rminus[1]**2)/(2.0*delta[i])

	return E, g

# ADAM optimizer
def adam_gradient(count,m,v,g):
		## This function implements adam optimizer
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
		alpha = [0.1 for i in range(len(g))]
		if count == 0:
			count = 1

		shift = [0 for i in range(len(g))]

		for i in range(len(g)):
			m[i] = beta1 * m[i] + (1 - beta1) * g[i]
			v[i] = beta2 * v[i] + (1 - beta2) * np.power(g[i],2)

			alpha[i] = alpha[i] * np.sqrt(1 - np.power(beta2,count)) / (1 - np.power(beta1,count))

			shift[i] = alpha[i]*(m[i]/(np.sqrt(v[i])+eps))

		return shift

#############################################

# VFK routine

def VFK(n_qubits=0, ancilla_qubits=0, operator=None,ansatz=None,
	ansatz_reps=None,init_params=None,instance=None,max_iter=100,opt='sgd',
	grad='param_shift', filename ='results.dat'):

	#This function is for the VQE procedure to find the ground state of the clock Hamiltonian

	#qubit_op    = StateFn(operator,is_measurement = True)
	qubit_op    = operator
	spins       =  n_qubits - ancilla_qubits
	nparameters = len(init_params)

	# initialize the useful quantities
	curr_params = init_params
	count       = 0

	E          = np.ones(2)
	g          = np.zeros((nparameters,2))
	g_norm     = 1
	g_norm_ths = nparameters*1e-5

	ths    = 0
	if opt == 'adam':
		m = np.zeros(nparameters)
		v = np.zeros(nparameters)

	#############################################
	# Initialize circuits and samplers
	params_vec      = ParameterVector('θ',nparameters)
	circ_wfn        = ansatz(n_qubits,ancilla_qubits,ansatz_reps,params_vec)

	print("The wavefunction ansatz is")
	print(circ_wfn)

	state_wfn       = StateFn(operator,is_measurement=True) @ StateFn(circ_wfn)

	## initialize useful quantities once
	if(instance.is_statevector):
		expectation = AerPauliExpectation()
	if(not instance.is_statevector): 
		expectation = PauliExpectation()
	
	sampler = CircuitSampler(instance)

	## Now prepare the state in order to compute the overlap and its gradient
	
	state_wfn = expectation.convert(state_wfn)



	# Create container for the final results

	energies = []
	params   = [list(init_params)]

	print("\nStarting the VQE algorithm")
	print("------------------------------------\n")
	while E[0] > ths and count < max_iter:# and g_norm > g_norm_ths:

		count = count +1

		if grad == 'param_shift':
			print("Using parameter shift rule")
			E, g = energy_and_gradient(state_wfn,curr_params,params_vec,expectation,sampler)
		if grad == 'spsa':
			print("Using SPSA")
			E, g = energy_and_gradient_spsa(state_wfn,curr_params,params_vec,expectation,sampler,count)

		print("Step: "+str(count))
		print("Energy: "+str(E[0]))
		print("Gradient:")
		print(g[:,0])
		

		energies.append(list(E))

		#Norm of the gradient
		g_vec = np.asarray(g[:,0])
		g_norm = np.linalg.norm(g_vec)



		if opt == 'adam':
			print("Optimizer: Adam \n")
			g_vec  = np.asarray(g[:,0])
			adam_g = adam_gradient(count,m,v,g_vec)

			curr_params = curr_params - adam_g

		elif opt == 'sgd' :
			print("Optimizer: SGD\n")
			curr_params = curr_params - g[:,0]

		if E[0] > ths and count < max_iter: # and g_norm > g_norm_ths:
			params.append(list(curr_params))

		print("============================\n")
	# Save data in an external file


	print("\nEnergy list: ")
	energies = np.asarray(energies)
	print(list(energies[:,0]))

	log_data = {}
	log_data["energies"]       = list(energies[:,0])
	log_data["energies_error"] = list(energies[:,1])
	log_data["parameters"]     = list(params)
	log_data["tot_steps"]      = [count]
	log_data["spins"]          = [spins]
	log_data["ancilla_qubits"] = [ancilla_qubits]
	log_data["ansatz_reps"]    = [ansatz_reps]


	json.dump(log_data, open( filename,'w+'))


#############################################

# We need functions to measure observables on the clock states created with the VQE


def measure_observable(obs,ancilla,time,ansatz,reps,params,instance):
	
	# Args:

	# - obs     : a Pauli operator to measure in the system qubits
	# - ancilla : the number of ancilla qubits
	# - time    : at which time the observable has to be measured
	

	## Create the ancilla projector

	gray_string = gray_code(ancilla)[time]

	for i in range(len(gray_string)):

		if i == 0:
			if gray_string[i] == '0':
				prj = (I+Z)
			if gray_string[i] == '1':
				prj = (I-Z)
		
		if i > 0:
			if gray_string[i] == '0':
				prj = prj^(I+Z)
			if gray_string[i] == '1':
				prj = prj^(I-Z)
	

	## Now that we have the projector, compose with the observables
	coeff = float(1/np.power(2,int(ancilla)))
	prj = coeff*prj
	
	obs = SparsePauliOp.from_operator(obs)
	obs = PauliSumOp(obs)


	tot_obs = obs ^ prj
	## Create the evaluation circuit
	
	op  = StateFn(tot_obs,is_measurement = True)
	wfn = StateFn(ansatz(op.num_qubits,ancilla,reps,params))
	

	# Evaluate the aux operator given
	braket = op @ wfn

	if(instance.is_statevector):
		expectation = AerPauliExpectation()
	if(not instance.is_statevector): 
		expectation = PauliExpectation()

	grouped    = expectation.convert(braket)
	sampled_op = CircuitSampler(instance).convert(grouped)

	mean_value = np.power(2,int(ancilla))*sampled_op.eval().real
	est_err = 0

	if (not instance.is_statevector):
		shots    = instance.run_config.shots
		variance = PauliExpectation().compute_variance(sampled_op).real
		est_err  = np.sqrt(variance/shots)

	res = [mean_value,est_err]

	return res


def measure_observable_binary(obs,ancilla,time,ansatz,reps,params,instance):
	
	# Args:

	# - obs     : a Pauli operator to measure in the system qubits
	# - ancilla : the number of ancilla qubits
	# - time    : at which time the observable has to be measured
	

	## Create the ancilla projector

	gray_string = binary_code(ancilla)[time]

	for i in range(len(gray_string)):

		if i == 0:
			if gray_string[i] == '0':
				prj = (I+Z)
			if gray_string[i] == '1':
				prj = (I-Z)
		
		if i > 0:
			if gray_string[i] == '0':
				prj = prj^(I+Z)
			if gray_string[i] == '1':
				prj = prj^(I-Z)
	

	## Now that we have the projector, compose with the observables
	coeff = float(1/np.power(2,int(ancilla)))
	prj = coeff*prj
	
	obs = SparsePauliOp.from_operator(obs)
	obs = PauliSumOp(obs)


	tot_obs = obs ^ prj
	## Create the evaluation circuit
	
	op  = StateFn(tot_obs,is_measurement = True)
	wfn = StateFn(ansatz(op.num_qubits,ancilla,reps,params))
	

	# Evaluate the aux operator given
	braket = op @ wfn

	if(instance.is_statevector):
		expectation = AerPauliExpectation()
	if(not instance.is_statevector): 
		expectation = PauliExpectation()

	grouped    = expectation.convert(braket)
	sampled_op = CircuitSampler(instance).convert(grouped)

	mean_value = np.power(2,int(ancilla))*sampled_op.eval().real
	est_err = 0

	if (not instance.is_statevector):
		shots    = instance.run_config.shots
		variance = PauliExpectation().compute_variance(sampled_op).real
		est_err  = np.sqrt(variance/shots)

	res = [mean_value,est_err]

	return res











