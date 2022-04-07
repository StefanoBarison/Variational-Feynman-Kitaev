using Yao,YaoExtensions,PyCall,YaoPlots, JSON, Statistics, PlutoUI, Compose, LaTeXStrings, Yao.AD, LinearAlgebra

using QuAlgorithmZoo: Adam, update!

# include all the useful functions for VQE
include("vfk_functions.jl")


## Now we can declare all the important quantities for the system

n_spins   = 2
n_ancilla = 3
J         = 0.25
B         = 1.0
dt_vec    = Dict(2=>1.0,3=>0.428,4=>0.2,5=>0.096,6=>0.047,7=>0.0236)
dt        = dt_vec[n_ancilla]

## For ansatz
depth     = 2
n_qubits  = n_spins + n_ancilla

## Now we want to create the ansatz

n_params  = 2*depth*n_ancilla*(n_spins^2+n_spins)
q_circ = example_ansatz(n_qubits,n_ancilla,depth,zeros(n_params))


# To plot it
#YaoPlots.plot(q_circ) |> SVG("ansatz_circ.svg")


## Choose how to vary U(f(k,t)) (see Appendix A)

# kth-root 
#k_steps   = vcat([100 - i*5 for i in 0:19],[5 - i for i in 1:4])
# incremental
k_steps    = vcat([(1/100)*i for i in 1:5:99],[1])
# single              
#k_steps   = [1]


optimizer = Adam(lr=0.01) 
params    = parameters(q_circ)
energies  = []
grad_mod  = []
it        = 200
final_it  = 3000


println("VQE")
global low_energy = 1
for (k,k_step) in enumerate(k_steps)

	# Create the Hamiltonian for this k
	println("Step: "*string(k))
	dt_k = dt*k_step

	## Interaction
	# Binary encoding
	#ham_jl_kron_k = create_clock_hamiltonian(n_spins,n_ancilla,dt_k,J,B)
	# Gray encoding
	ham_jl_kron_k = create_clock_hamiltonian_gray(n_spins,n_ancilla,dt_k,J,B)
	println("Hamiltonian created")

	# optimizer = Adam(lr=0.01)

	curr_it = 1
	if k_step == 1
		curr_it = final_it
		global optimizer = Adam(lr=0.01,beta1=0.9,beta2=0.999)
			
	else 
		curr_it = it
	end

	for i = 1:curr_it
		## `expect'` gives the gradient of an observable.
		grad_input, grad_params = expect'(ham_jl_kron_k, zero_state(n_qubits) => q_circ)
		push!(grad_mod,norm(grad_params))
		## feed the gradients into the circuit.
			
		## ADAM
		dispatch!(q_circ, update!(params, grad_params, optimizer))

		## SGD
		#η  = 0.05
		#up_params = params - grad_params.*η
		#dispatch!(q_circ, up_params)
		#global params = up_params

		## Save the new energy
		local new_energy = real.(expect(ham_jl_kron_k, zero_state(n_qubits) |> q_circ))
		println("Step $i, Energy = $new_energy")
		push!(energies,new_energy)

		if k_step == 1 && new_energy < low_energy
			global low_params = parameters(q_circ)
			global low_energy = new_energy
		end
	end
end
final_params = parameters(q_circ)


# Now we want to save the results obtained
save_data = true


if save_data
	res = Dict("final_params"=>final_params,"low_params"=>low_params,"low_energy"=>[low_energy],"energies"=> energies,"ansatz_reps"=>[depth],"spins"=> [n_spins], "ancilla_qubits"=> [n_ancilla],"dt"=>[dt])
	j_res = JSON.json(res)

	open("data/VFK/VFK_depth"*string(depth)*"_"*string(n_spins)*"spin_"*string(n_ancilla)*"ancilla_dt"*string(dt)*"_gray.dat","w") do j
		write(j,j_res)
	end
	
end


