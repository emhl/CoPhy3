using Statistics: mean, std
using StatsBase: autocor
using HDF5
using DelimitedFiles

"Compute the sum of the neighboring spins of (i, j, k)"
function nn(x::Array, i::Number, j::Number)
    result  = x[mod(i-2, L)+1, j] + x[mod(i, L)+1, j]
    result += x[i, mod(j-2, L)+1] + x[i, mod(j, L)+1]
end

"Wolff step"
function move!(x::Array, ME::Array, p::Number)
    # randomly pick a spin
    irn = rand(1:L)
    jrn = rand(1:L)
    # flip spin and neighbors if they are in the same cluster
    flip!(x, ME, p, irn, jrn)
end

"flip spin and neighbors if they are in the same cluster"
function flip!(x::Array, ME::Array, p::Number, i::Number, j::Number)
    spin = x[i, j]
    x[i, j] = -spin

    # update magnetization and energy
    ME[1] -= 2*spin
    ME[2] += 2*J*spin*nn(x, i, j)

    # add alligned neighbors to the cluster with probability p and flip! them
    check_and_flip!(x, ME, p, mod(i-2, L)+1, j, spin)
    check_and_flip!(x, ME, p, i, mod(j-2, L)+1, spin)
    check_and_flip!(x, ME, p, mod(i, L)+1, j, spin)
    check_and_flip!(x, ME, p, i, mod(j, L)+1, spin)
end

"add alligned spins to the cluster with probability p and flip! them"
function check_and_flip!(x::Array, ME::Array, p::Number, i::Number, j::Number, spin::Number)
    # flip neighbors with probability p if spins alligned
    if x[i, j] == spin && rand()<p
        flip!(x, ME, p, i, j)
    end
end

#= Global parameters =#
L = 32                                # system size
J = 1                                # spin coupling strength
T = [1.0,2.2,3.0] # inverse temperature
Nthermalization = 100              # number of thermalization steps
Nsample = 1000                   # number of observation steps

magnetization  = zeros(size(T))   # average magnetization
energy         = zeros(size(T))   # average energy
M_std          = zeros(size(T))   # standard deviation of the magnetization
E_std          = zeros(size(T))   # standard deviation of the energy
susceptibility = zeros(size(T))   # magnetic susceptibility
heatcapacity   = zeros(size(T))   # heat capacity

h5open("ising_data_L"*string(L)*".h5", "w") do file
for (ib, t) in enumerate(T)
    println("Running for inverse temperature = $t")

    #= Initialization of a random spin configuration =#
    x = ones(L, L)       # all spins up
    M = L^2                 # corresponding magnetization
    E = -3*J*L^2            # corresponding energy
    for j=1:L, i=1:L # inverted order because julia is column major
        if rand() < 0.5
            x[i, j] = -1
            M -= 2
            E += 2*J*nn(x, i, j)
        end
    end
    ME = [M, E]             # array with magnetization and energy of x
    p = 1- exp(-2J/t)       # probability(add lattice site to cluster)

    #= Thermalization =#
    for _ in 1:Nthermalization
        move!(x, ME, p)
    end

    # Sampling
    M_arr = zeros(Nsample)
    E_arr = zeros(Nsample)

    configs = zeros(L, L, Nsample)
    for m in 1:Nsample
        move!(x, ME, p)
        M_arr[m] = abs(ME[1])
        E_arr[m] = ME[2]
        configs[:,:,m] = x
    end

    #= Post processing of the data =#
    M_arr /= L^2
    magnetization[ib]  = mean(M_arr)
    energy[ib]         = mean(E_arr)
    M_std[ib]          = std(M_arr)
    E_std[ib]          = std(E_arr)
    susceptibility[ib] = L^2*1/t*M_std[ib]^2
    heatcapacity[ib]   = (1/t)^2*E_std[ib]^2

    #= save configurations =#
    write(file, string(t), configs)
end
end

open("data_L"*string(L)*".txt", "w") do io
    writedlm(io, [magnetization susceptibility energy heatcapacity])
end
