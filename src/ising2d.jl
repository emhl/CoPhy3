using Statistics
using ProgressMeter

function create_grid(L::Int; up_prob::Float64=0.5)
    # up_prob is the probability of a spin being up
    grid = rand(L, L) .< up_prob # creates boolean grid
    grid = 2 * grid .- 1 # converts boolean grid to -1, 1 grid
    return grid
end


@doc "Energy calculation for different input arguments"
function grid_energy(grid::Array{Int,2}, J::Float64, B::Float64=0.0)
    N1, N2 = size(grid)
    E = 0.0
    for i in 1:N1
        for j in 1:N2
            E += -J * grid[i, j] * (
                     grid[mod1(i + 1, N1), j]
                     +
                     grid[i, mod1(j + 1, N2)]
                 )
        end
    end
    E -= B * sum(grid)
    return E
end

# create function that only calculates the energy diference that the single flip would cause
function energy_diff(grid::Array{Int,2}, flip_position::Tuple{Int,Int}; J::Float64=1.0, B::Float64=0.0)
    N1, N2 = size(grid)
    delta_E = 0.0
    i, j = flip_position
    proposed_spin = -grid[i, j]
    # 2x because the Energy contribution is symmetric
    delta_E += -J * 2 * proposed_spin * (
                   grid[mod1(i + 1, N1), j]
                   + grid[i, mod1(j + 1, N2)]
                   + grid[mod1(i - 1, N1), j]
                   + grid[i, mod1(j - 1, N2)]
               )
    delta_E -= B * 2 * proposed_spin
    return delta_E
end

@doc "Creates a lookup table for the values of the exponential function."
function create_lookup_table(T::Float64; J::Float64=1.0, dimension::Int=2)
    lookup_table = Dict{Float64,Float64}()
    for e in -4*dimension:4*dimension
        lookup_table[J*e] = state_probability(J * e, T)
    end
    return lookup_table
end

@doc "Magnetisation of a given grid"
function grid_magnetisation(grid::Array{Int,2})
    return sum(grid) / length(grid)
end

@doc "Calculates the probabilites for a given temperature and Energy for the lookup table."
function state_probability(E::Float64, T::Float64)
    return exp(-E / T)
end


@doc "Metropolitan Step function for different input arguments"
function metropolis_step(grid::Array{Int,2}, J::Float64, lookup_table::Dict{Float64,Float64}, T::Float64=0.0, B::Float64=0.0)
    N1, N2, = size(grid)
    i, j = rand(1:N1), rand(1:N2)
    dE = energy_diff(grid, (i, j), J=J, B=B)
    if rand() < lookup_table[dE]
        grid[i, j] *= -1
        dM = 2 * grid[i, j]
    else
        dE = 0
        dM = 0
    end
    return grid, dE, dM
end

function cluster_edge_sum(grid::Array{Int,2}, cluster::Array{Bool,2})
    N1, N2, size(grid)
    edge_sum = 0
    for c in findall(cluster)
        neighbours = [(mod1(c[1] + 1, N1), c[2]),
            (c[1], mod1(c[2] + 1, N2)),
            (mod1(c[1] - 1, N1), c[2]),
            (c[1], mod1(c[2] - 1, N2))]
        for n in neighbours
            if !cluster[n...]
                edge_sum += grid[n...]
            end
        end
    end
    return edge_sum
end

function wolff_cluster(grid::Array{Int,2}, position::Tuple{Int,Int}, J::Float64, lookup_table::Dict{Float64,Float64})
    N1, N2 = size(grid)
    i, j = position

    state = grid[i, j]
    cluster = zeros(Bool, N1, N2)
    cluster[i, j] = true

    stack = [(i, j)]
    while !isempty(stack)
        i, j = pop!(stack)
        neighbours = [(mod1(i + 1, N1), j),
            (i, mod1(j + 1, N2)),
            (mod1(i - 1, N1), j),
            (i, mod1(j - 1, N2))]
        for n in neighbours
            neighbour_state = grid[n...]
            if !cluster[n...] && neighbour_state == state && rand() < 1 - lookup_table[2*J]
                cluster[n...] = true
                push!(stack, n)
            end
        end
    end
    return cluster, state
end


@doc "Wolff Monte Carlo algorithm for ising model"
function wolff_step(grid::Array{Int,2}, J::Float64, lookup_table::Dict{Float64,Float64}, T::Float64=0.0, B::Float64=0.0)
    N1, N2 = size(grid)
    i, j = rand(1:N1), rand(1:N2)

    cluster, state = wolff_cluster(grid, (i, j), J, lookup_table)

    dE = 2 * J * cluster_edge_sum(grid, cluster) * state
    dM = -2 * sum(cluster) * state

    grid[cluster] .= -state

    return grid, dE, dM
end



@doc "Function to create an equilibrated grid after N Thermalisation steps"
function create_equilibrated_grid(; grid_size::Int=10, J::Float64=1.0, lookup_table::Dict{Float64,Float64}, T::Float64=0.0, B::Float64=0.0, N::Int=100 * grid_size^3, initial_up_prob::Float64=0.5, mc_algorithm::Function=metropolis_step)
    grid = create_grid(grid_size, up_prob=initial_up_prob) # always start with a new random grid
    for i in 1:N
        grid, _ = mc_algorithm(grid, J, lookup_table, T, B)
    end
    return grid
end

function subsweep(grid::Array{Int,2}, lookup_table::Dict{Float64,Float64}; J::Float64=1.0, T::Float64=0.0, B::Float64=0.0, N::Int=1_000, mc_algorithm::Function=metropolis_step)
    dE, dM = 0.0, 0.0
    for i in 1:N
        grid, dE_, dM_ = mc_algorithm(grid, J, lookup_table, T, B)
        dE += dE_
        dM += dM_
    end
    return grid, dE, dM
end


function sample_grid(grid::Array{Int,2}, lookup_table::Dict{Float64,Float64}; J::Float64=1.0, T::Float64=0.0, B::Float64=0.0, N::Int=1_000, N_Subsweep::Int=1_000, mc_algorithm::Function=metropolis_step)
    energies, magnetisations = Vector{Float64}(undef, N), Vector{Float64}(undef, N)
    energy_ = grid_energy(grid, J, B)
    magnetisation_ = grid_magnetisation(grid)
    grid_len = length(grid)
    grids = Array{Array{Int,1},1}(undef, N)
    for i in 1:N
        grid, dE, dM = subsweep(grid, lookup_table, J=J, T=T, B=B, N=N_Subsweep, mc_algorithm=mc_algorithm)
        energy_ += dE
        magnetisation_ += dM / grid_len
        energies[i] = energy_
        magnetisations[i] = magnetisation_
        grids[i] = reshape(grid, grid_len)
    end
    return energies, magnetisations, grids
end

function measure_single_config(; grid_size::Int=10, J::Float64=1.0, T::Float64=0.0, B::Float64=0.0, N_Sample::Int=1000, N_Thermalize::Int=100 * grid_size^2, N_Subsweep::Int=3 * grid_size^2, initial_up_prob::Float64=0.5, mc_algorithm::Function=metropolis_step)
    lookup_table = create_lookup_table(T, J=J)
    grid = create_equilibrated_grid(grid_size=grid_size, J=J, lookup_table=lookup_table, T=T, B=B, N=N_Thermalize, initial_up_prob=initial_up_prob, mc_algorithm=mc_algorithm)
    energies, magnetisations = sample_grid(grid, lookup_table,J=J, T=T, B=B, N=N_Sample, N_Subsweep=N_Subsweep, mc_algorithm=mc_algorithm)
    # only take the absolute value of the magnetisation, because the system is symmetric
    return (mean(energies), std(energies), mean(energies .^ 2), mean(energies .^ 4)), (mean(abs.(magnetisations)), std(abs.(magnetisations)), mean(abs.(magnetisations) .^ 2), mean(abs.(magnetisations) .^ 4))
end

@doc "function for sweeping over a temperature intervall using T_Steps steps"
function temp_sweep(; grid_size::Int=10, J::Float64=1.0, T_Start::Float64=0.0, T_End::Float64=10.0, B::Float64=0.0, T_Steps::Int=100, N_Sample::Int=1000, N_Thermalize::Int=100 * grid_size^3, N_Subsweep::Int=3 * grid_size^3, initial_up_prob::Float64=0.5, mc_algorithm::Function=metropolis_step)
    energies, energies_std, energies_2, energies_4 = Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps)
    magnetisations, magnetisations_std, magnetisations_2, magnetisations_4 = Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps)
    temps = Vector{Float64}(undef, T_Steps)
    @showprogress Threads.@threads for (iT, T) in collect(enumerate(range(T_Start, T_End, T_Steps)))
        (energies[iT], energies_std[iT], energies_2[iT], energies_4[iT]), (magnetisations[iT], magnetisations_std[iT], magnetisations_2[iT], magnetisations_4[iT]) = measure_single_config(grid_size=grid_size, J=J, T=T, B=B, N_Sample=N_Sample, N_Thermalize=N_Thermalize, N_Subsweep=N_Subsweep, initial_up_prob=initial_up_prob, mc_algorithm=mc_algorithm)
        temps[iT] = T
    end
    return (energies, energies_std, energies_2, energies_4), (magnetisations, magnetisations_std, magnetisations_2, magnetisations_4), temps
end

function simple_monte_carlo(; grid_size::Int=10, J::Float64=1.0, T::Float64=0.0, B::Float64=0.0, N::Int=100_000, N_Subsweep::Int=1, initial_up_prob::Float64=0.5, mc_algorithm::Function=metropolis_step)
    grid = create_grid(grid_size, up_prob=initial_up_prob)
    energies, magnetisations = Vector{Float64}(undef, N), Vector{Float64}(undef, N)
    lookup_table = create_lookup_table(T, J=J)
    E = grid_energy(grid, J, B)
    M = grid_magnetisation(grid)
    for i in 1:N
        for j in 1:N_Subsweep
            grid, dE, dM = mc_algorithm(grid, J, lookup_table, T, B)
            E += dE
            M += dM
        end
        energies[i] = E
        magnetisations[i] = M
    end
    return energies, magnetisations
end

@doc "derivative of one vector by the other"
function dv(x::Vector, y::Vector)
    if length(y) > length(x)
        laenge = length(x)
    else
        laenge = length(y)
    end

    derivative = Float64[]

    for i in 1:laenge
        if i == 1
            push!(derivative, (y[i+1] - y[i]) / (x[i+1] - x[i]))
        elseif i == laenge
            push!(derivative, (y[i] - y[i-1]) / (x[i] - x[i-1]))
        else
            push!(derivative, (y[i+1] - y[i-1]) / (x[i+1] - x[i-1]))
        end
    end

    return derivative
end

@doc "autocorrelation function"
function autocorr(x::Vector{Float64}; max_lag::Int=100)
    n = length(x)
    x_m = mean(x)
    x_ = x .- x_m
    r = zeros(max_lag)
    Threads.@threads for k in 1:max_lag
        r[k] = sum(x_[1:(n-k)] .* x_[(k+1):n]) / (n - k)
    end
    return r
end
