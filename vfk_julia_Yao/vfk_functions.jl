#### This file contains the useful functions for VQE in Julia


### Useful Gates

function Rzz(n,i,j,theta)
	circ = chain(n,[cnot(i,j),put(j=>Rz(theta)),cnot(i,j)])
	return circ
end


### Custom Gate

function custom_gate(n,i,j,param)
	circ = chain(n, [put(j=>Ry(param[1])),put(j=>Rx(param[2])),cnot(i,j),put(j=>Rx(param[3])),put(j=>Ry(param[4])),cnot(i,j)])
	return circ
end


### Ansatz

function example_ansatz(n,ancille,depth,params)
	count = 1
	ent_list= [(l,m) for l in (ancille+1):n for m in l+1:n]
	circ  = chain(n, put(i=> H) for i in 1:ancille)

	for d in 1:depth
		for i in 1:ancille
			for j in (ancille+1):n
				push!(circ,custom_gate(n,i,j,params[count:count+3]))
				count = count+4
			end

			for (l,m) in ent_list
				push!(circ,custom_gate(n,l,m,params[count:count+3]))
				count = count+4
			end

		end
	end
	return circ
end


#############################################

# Function to write gray codes

grayencode(n::Integer) = n ⊻ (n >> 1)

function graydecode(n::Integer)
    r = n
    while (n >>= 1) != 0
        r ⊻= n
    end
    return r
end

function gray_ancilla(n)
	gray_array = []

	for i in 0:(2^n)-1
		push!(gray_array,digits(grayencode(i), base=2, pad=n) |> reverse)
	end
	return gray_array
end

function gray_state(n,time)
	gray_string = digits(grayencode(time), base=2, pad=n) |> reverse

	circ = chain(n)

	for i in 1:n
		if gray_string[i] == 1
			push!(circ,put(i=>X))
		end
	end
	return circ
end


### Functions to write ancilla in binaries

function binary_ancilla(n)
	bin_array = []

	for i in 0:(2^n)-1
		push!(bin_array,digits(i, base=2, pad=n) |> reverse)
	end
	return bin_array
end



#############################################

### Projectors

function projector_zero(n_qubits)

	prj_list = [0.5*(I2+Z) for i in 1:n_qubits]
	prj = kron(prj_list...)

	return prj
end

function projector_one(n_qubits)

	prj_list = [0.5*(I2-Z) for i in 1:n_qubits]
	prj = kron(prj_list...)

	return prj
end

function projector_time(n_qubits,time)
	t_digit = digits(time, base=2, pad=n_qubits)  |> reverse

	prj_list = []
	for i in 1:length(t_digit)
		if t_digit[i] == 0
			push!(prj_list,0.5*(I2+Z))
		elseif t_digit[i] == 1
			push!(prj_list,0.5*(I2-Z))
		end
	end

	prj = kron(prj_list...)
	return prj
end

function projector_time_gray(n_qubits,time)
	gray = digits(grayencode(time), base=2, pad=n_qubits)  |> reverse
	
	prj_list = []
	for i in 1:length(gray)
		if gray[i] == 0
			push!(prj_list,0.5*(I2+Z))
		elseif gray[i] == 1
			push!(prj_list,0.5*(I2-Z))
		end
	end
	
	prj = kron(prj_list...)
	return prj
end


#############################################

### Create Hamiltonian

# Binary encoding
function create_clock_hamiltonian(n_spin,n_ancilla,dt,J,B)
	# First, create the initial projector term
	id_spin    = kron([I2 for i in 1:n_spin]...)
	id_ancilla = kron([I2 for i in 1:n_ancilla]...)
	id_tot     = kron([I2 for i in 1:(n_ancilla+n_spin)]...)

	## C0

	C0 = kron(projector_zero(n_ancilla),id_spin-projector_zero(n_spin))

	# C1

	C1 = 0.5*kron(projector_time(n_ancilla,0),id_spin) + 0.5*kron(projector_time(n_ancilla,(2^n_ancilla)-1),id_spin)

	for i in 1:(2^n_ancilla) - 2
		C1 += kron(projector_time(n_ancilla,i),id_spin)
	end


	# C2

	# - The Trotter approximation of U(dt)
	Udt = cos(J*dt)*id_spin-im*sin(J*dt)*kron(n_spin,1=>Z,2=>Z)

	for j in 2:n_spin-1
		Udt = Udt*(cos(J*dt)*id_spin-im*sin(J*dt)*kron(n_spin,j=>Z,j+1=>Z))
	end
	#Udt = id_spin
	for j in 1:n_spin
			Udt = Udt*(cos(B*dt)*id_spin-im*sin(B*dt)*kron(n_spin,j=>X))
	end

	Udt_dag = adjoint(Udt)

	# - Excitation operator for ancillae and total operator
	bin_list  = binary_ancilla(n_ancilla)
	C2_list    = []

	for j in 1:(2^n_ancilla)-1


		# Find out which bit changes
		# in binary mode more bits changes

		exc_list = []
		for k in 1:length(bin_list[j])
            if bin_list[j][k] - bin_list[j+1][k] == -1
                push!(exc_list,0.5*(X+im*Y)) #len(g1)-(g+1) #to reverse
            elseif bin_list[j][k] - bin_list[j+1][k] == +1
            	push!(exc_list,0.5*(X-im*Y))
            elseif bin_list[j][k] - bin_list[j+1][k] == 0 && bin_list[j][k] == 0
            	push!(exc_list,0.5*(I2+Z))
            elseif bin_list[j][k] - bin_list[j+1][k] == 0 && bin_list[j][k] == 1
            	push!(exc_list,0.5*(I2-Z))
			end
		end

		exc_list = exc_list |> reverse
		exc_op = kron(exc_list...)
		exc_op_dag = adjoint(exc_op)

		push!(C2_list,kron(exc_op_dag,Udt) + kron(exc_op,Udt_dag))
	end


	C2 = C2_list |> sum

	C = C0 + C1 -0.5*C2
	
	return C

end

# Gray encoding
function create_clock_hamiltonian_gray(n_spin,n_ancilla,dt,J,B)
	# First, create the initial projector term
	id_spin    = kron([I2 for i in 1:n_spin]...)
	id_ancilla = kron([I2 for i in 1:n_ancilla]...)
	id_tot     = kron([I2 for i in 1:(n_ancilla+n_spin)]...)
	
	C0 = kron(projector_zero(n_ancilla),id_spin-projector_zero(n_spin))
	
	# Now, the local projector terms
	
	C1 = 0.5*kron(projector_time_gray(n_ancilla,0),id_spin) + 0.5*kron(projector_time_gray(n_ancilla,(2^n_ancilla)-1),id_spin)
	
	for i in 1:(2^n_ancilla) - 2
		C1 += kron(projector_time_gray(n_ancilla,i),id_spin)
	end
	
	
	# Then, the time evolution terms:
	
	# - The Trotter approximation of U(dt)
	Udt = cos(J*dt)*id_spin-im*sin(J*dt)*kron(n_spin,1=>Z,2=>Z)
	
	for j in 2:n_spin-1
		Udt = Udt*(cos(J*dt)*id_spin-im*sin(J*dt)*kron(n_spin,j=>Z,j+1=>Z))
	end
	
	for j in 1:n_spin
			Udt = Udt*(cos(B*dt)*id_spin-im*sin(B*dt)*kron(n_spin,j=>X))
	end
	
	Udt_dag = adjoint(Udt)
	
	# - Excitation operator for ancillae and total operator
	gray_list  = gray_ancilla(n_ancilla)
	C2_list    = []
	
	for j in 1:(2^n_ancilla)-1
		
		
		# Find out which bit changes
        exc_site = 0
        for k in 1:length(gray_list[j])
            if gray_list[j][k] != gray_list[j+1][k]
                exc_site = k #len(g1)-(g+1) #to reverse
			end
		end
		
		# Compose the operator
		exc_list = []
		for i in 1:length(gray_list[j+1])
			
			if     i == exc_site && gray_list[j][exc_site] == 0
				push!(exc_list,0.5*(X+im*Y))
			elseif i == exc_site && gray_list[j][exc_site] == 1
				push!(exc_list,0.5*(X-im*Y))
			elseif i != exc_site && gray_list[j+1][i] == 0
				push!(exc_list,0.5*(I2+Z))
			elseif i != exc_site && gray_list[j+1][i] == 1
				push!(exc_list,0.5*(I2-Z))
			end
		end
		
		exc_op = kron(exc_list...)
		exc_op_dag = adjoint(exc_op)
		
		push!(C2_list,kron(exc_op_dag,Udt) + kron(exc_op,Udt_dag))
	end
		
	
	C2 = C2_list |> sum
	
	C = C0 + C1 -0.5*C2
	return C
	
end


#############################################

### For clock VQE measuring observables

function measure_observable(obs,ancilla,time,ansatz,reps,params)

	#=
	 Args:

	- obs     : a Pauli operator to measure in the system qubits
	- ancilla : the number of ancilla qubits
	- time    : at which time the observable has to be measured
	=#

	## Create the ancilla projector

	bin_string = binary_ancilla(ancilla)[time] |> reverse

	prj = kron(map(bin_string) do t
		if t == 0
			return (I2+Z)
		else
			return (I2-Z)
		end
		end...)

	## Now that we have the projector, compose with the observables
	coeff = 1/(2^ancilla)
	prj = coeff*prj

	tot_obs = kron(prj,obs) # Note that this is an inverse order wrt Qiskit
	## Create the evaluation circuit
	n_qubits = nqubits(tot_obs)
	dispatch!(ansatz, params)

	res = (2^ancilla)*expect(tot_obs,zero_state(n_qubits) |> ansatz)

	return res
end

function measure_observable_gray(obs,ancilla,time,ansatz,reps,params)
	
	#=
	 Args:

	- obs     : a Pauli operator to measure in the system qubits
	- ancilla : the number of ancilla qubits
	- time    : at which time the observable has to be measured
	=#

	## Create the ancilla projector

	gray_string = gray_ancilla(ancilla)[time]

	prj = kron(map(gray_string) do t
		if t == 0
			return (I2+Z)
		else
			return (I2-Z)
		end
		end...)	

	## Now that we have the projector, compose with the observables
	coeff = 1/(2^ancilla)
	prj = coeff*prj
	
	tot_obs = kron(prj,obs) # Note that this is an inverse order wrt Qiskit			
	## Create the evaluation circuit
	n_qubits = nqubits(tot_obs)
	dispatch!(ansatz, params)
	
	res = (2^ancilla)*expect(tot_obs,zero_state(n_qubits) |> ansatz)

	return res
end

#############################################

