using Yao,YaoExtensions,PyCall,YaoPlots, JSON, Statistics, LaTeXStrings

# include all the useful functions for VQE
include("vfk_functions.jl")

import PyPlot; const plt = PyPlot

# Now import useful subroutines from python
pushfirst!(PyVector(pyimport("sys")."path"), "")

## Sys data
n_spins   = 2
n_ancilla = 3
n_qubits  = n_spins + n_ancilla
depth     = 2
dt_vec    = Dict(2=>1.0,3=>0.428,4=>0.2,5=>0.096,6=>0.047,7=>0.0236)
dt        = dt_vec[n_ancilla]

## import data from JSON
# Binary
#data     = JSON.parse(open("data/VFK/VFK_depth"*string(depth)*"_"*string(n_spins)*"spin_"*string(n_ancilla)*"ancilla_dt"*string(dt)*".dat","r"))
# Gray
data     = JSON.parse(open("data/VFK/VFK_depth"*string(depth)*"_"*string(n_spins)*"spin_"*string(n_ancilla)*"ancilla_dt"*string(dt)*"_gray.dat","r"))


final_energy = data["low_energy"]
params       = data["low_params"] 

q_circ = example_ansatz(n_qubits,n_ancilla,depth,params)

## Now we will measure the observables using the parameters obtained

obs = kron(n_spins,1=>Z)
o_vec = []
for i in 1:2^n_ancilla
	#Binary
	#push!(o_vec, real.(measure_observable(obs,n_ancilla,i,q_circ,depth,params)))
	# Gray
	push!(o_vec, real.(measure_observable_gray(obs,n_ancilla,i,q_circ,depth,params)))
end


#Interaction
ex_res = JSON.parse(open("data/Exact/OBS_exact_ising_"*string(n_spins)*"spin_J0.25_B1.dat","r"))

#close("all")
plt.title(string(n_spins)*" spins and "*string(n_ancilla)*" ancillae, dt = "*string(dt))
plt.plot(ex_res["times"][1:61],ex_res["Sz_0"][1:61],linestyle="dashed",color="black",label= "Exact")
plt.plot([i*dt for i in 0:(2^n_ancilla)-1],o_vec,linestyle="",marker="o",label = "VQE - depth = "*string(depth))
plt.xlabel("t")
plt.ylabel(L"$ \langle \sigma_{1}^{z} \rangle $")
plt.legend()
plt.gcf()