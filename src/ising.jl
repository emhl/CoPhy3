using Statistics
using ProgressMeter

function create_grid(N1::Int, N2::Int=N1, N3::Int=N1; up_prob::Float64=0.5)
    # up_prob is the probability of a spin being up
    grid = rand(N1, N2, N3) .< up_prob # creates boolean grid
    grid = 2 * grid .- 1 # converts boolean grid to -1, 1 grid
    return grid
end

@doc "Cluster labeling algorithm for neighbouring spins"
function hoshen_kopelmann_clustering(grid::Array{Int,3})
    # variation of the Hoshen-Kopelmann algorithm, because there are no unoccupied sites
    N1, N2, N3 = size(grid)
    labels = zeros(Int, N1, N2, N3)
    next_label = 1
    for i in 1:N1, j in 1:N2, k in 1:N3
        neighbours = [(mod1(i - 1, N1), j, k), (i, mod1(j - 1, N2), k), (i, j, mod1(k - 1, N3))] # periodic boundary conditions

        state = grid[i, j, k]
        neighbour_states = [grid[n...] for n in neighbours]
        neighbours = neighbours[neighbour_states.==state]

        neighbour_labels = [labels[n...] for n in neighbours]
        if sum(neighbour_labels) == 0
            labels[i, j, k] = next_label
            next_label += 1
        else
            neighbour_labels = neighbour_labels[neighbour_labels.!=0] # remove 0 labels
            labels[i, j, k] = minimum(neighbour_labels) # assign the minimum label
            for l in neighbour_labels
                labels[labels.==l] .= labels[i, j, k] # relabel the neighbours
            end
        end
    end
    return labels
end

@doc "Energy calculation for different input arguments"
function energy(grid::Array{Int,3}, J::Float64, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    E = 0.0
    for i in 1:N1
        for j in 1:N2
            for k in 1:N3
                E += -J * grid[i, j, k] * (
                         grid[mod1(i + 1, N1), j, k]
                         + grid[i, mod1(j + 1, N2), k]
                         + grid[i, j, mod1(k + 1, N3)]
                     )
            end
        end
    end
    E -= B * sum(grid)
    return E
end

# create function that only calculates the energy diference that the single flip would cause
function energy_diff(grid::Array{Int,3}, flip_position::Tuple{Int,Int,Int}; J::Float64=1.0, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    delta_E = 0.0
    i, j, k = flip_position
    proposed_spin = -grid[i, j, k]
    # 2x because the Energy contribution is symmetric
    delta_E += -J * 2 * proposed_spin * (
                   grid[mod1(i + 1, N1), j, k]
                   + grid[i, mod1(j + 1, N2), k]
                   + grid[i, j, mod1(k + 1, N3)]
                   + grid[mod1(i - 1, N1), j, k]
                   + grid[i, mod1(j - 1, N2), k]
                   + grid[i, j, mod1(k - 1, N3)]
               )
    delta_E -= B * 2 * proposed_spin
    return delta_E
end

@doc "Creates a lookup table for the values of the exponential function."
function create_lookup_table(T::Float64; J::Float64=1.0, dimension::Int=3)
    lookup_table = Dict{Float64,Float64}()
    for e in -4*dimension:4*dimension
        lookup_table[J*e] = state_probability(J * e, T)
    end
    return lookup_table
end

@doc "Magnetisation of a given grid"
function magnetisation(grid::Array{Int,3})
    return sum(grid) / length(grid)
end

@doc "Calculates the probabilites for a given temperature and Energy for the lookup table."
function state_probability(E::Float64, T::Float64)
    return exp(-E / T)
end

@doc "Calculates the weighted mean value of an Array"
function mean_observable(values::Array{Float64,1}, weights::Array{Float64,1})
    #TODO can this function be vectorized?
    # return sum(values .* weights) / sum(weights)
    s = 0.0
    for (value, weight) in zip(values, weights)
        s += value * weight
    end
    return s / sum(weights) # normalize
end

@doc "Calculates the weighted mean of the squared array values"
function mean_observable_squared(values::Array{Float64,1}, weights::Array{Float64,1})
    #TODO can this function be vectorized?
    # return sum((values .^ 2) .* weights) / sum(weights)
    s = 0.0
    for (value, weight) in zip(values, weights)
        s += value^2 * weight
    end
    return s / sum(weights) # normalize
end

@doc "Calculates the magnetic suscebtibility using the variance of the magnetization"
function magnetic_suceptibility(magnetisation_values::Array{Float64,1}, weights::Array{Float64,1})
    return mean_observable_squared(magnetisation_values, weights) - mean_observable(magnetisation_values, weights)^2
end

@doc "Calculates the heat capacity using the variance of the Energy"
function heat_capacity(energy_values::Array{Float64,1}, weights::Array{Float64,1}, T::Float64)
    return (mean_observable_squared(energy_values, weights) - mean_observable(energy_values, weights)^2) / T^2
end


@doc "Metropolitan Step function for different input arguments"
function metropolis_step(grid::Array{Int,3}, J::Float64, lookup_table::Dict{Float64,Float64}, T::Float64=0.0, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    i = rand(1:N1)
    j = rand(1:N2)
    k = rand(1:N3)
    dE = energy_diff(grid, (i, j, k), J=J, B=B)
    if rand() < lookup_table[dE]
        grid[i, j, k] *= -1
        dM = 2 * grid[i, j, k]
    else
        dE = 0
        dM = 0
    end
    return grid, dE, dM
end



@doc "Function for n repetitions of metropolis_step, tracking the energy and magnetization for constant temperature and field"
function monte_carlo_const_temp(grid::Array{Int,3}, J::Float64, T::Float64, B::Float64, n::Int64)

    energies, magnetisations = Vector{Float64}(undef, n), Vector{Float64}(undef, n)
    lookup_table = create_lookup_table(T, J=J)
    for i in 1:n
        grid, _ = metropolis_step(grid, J, lookup_table, T, B)
        energies[i] = energy(grid, J, B)
        magnetisations[i] = magnetisation(grid)
    end
    return grid, energies, magnetisations

end

@doc "Function to create an equilibrated grid after N Thermalisation steps"
function create_equilibrated_grid(; grid_size::Int=10, J::Float64=1.0, lookup_table::Dict{Float64,Float64}, T::Float64=0.0, B::Float64=0.0, N::Int=100 * grid_size^3, initial_up_prob::Float64=0.5)
    grid = create_grid(grid_size, up_prob=initial_up_prob) # always start with a new random grid
    for i in 1:N
        grid, _ = metropolis_step(grid, J, lookup_table, T, B)
    end
    return grid
end

function subsweep(grid::Array{Int,3}, J::Float64, lookup_table::Dict{Float64,Float64}, T::Float64=0.0, B::Float64=0.0, N::Int=1_000)
    dE, dM = 0.0, 0.0
    for i in 1:N
        grid, dE_, dM_ = metropolis_step(grid, J, lookup_table, T, B)
        dE += dE_
        dM += dM_
    end
    return grid, dE, dM
end


function sample_grid(grid::Array{Int,3}, J::Float64, lookup_table::Dict{Float64,Float64}; T::Float64=0.0, B::Float64=0.0, N::Int=1_000, N_Subsweep::Int=1_000)
    energies, magnetisations = Vector{Float64}(undef, N), Vector{Float64}(undef, N)
    energy_ = energy(grid, J, B)
    magnetisation_ = magnetisation(grid)
    grid_len = length(grid)
    for i in 1:N
        grid, dE, dM = subsweep(grid, J, lookup_table, T, B, N_Subsweep)
        energy_ += dE
        magnetisation_ += dM / grid_len
        energies[i] = energy_
        magnetisations[i] = magnetisation_
    end
    return energies, magnetisations
end

@doc "function for sweeping over a temperature intervall using T_Steps steps"
function temp_sweep(; grid_size::Int=10, J::Float64=1.0, T_Start::Float64=0.0, T_End::Float64=10.0, B::Float64=0.0, T_Steps::Int=100, N_Sample::Int=1000, N_Thermalize::Int=100 * grid_size^3, N_Subsweep::Int=3 * grid_size^3, initial_up_prob::Float64=0.5)
    energies, energies_std, magnetisations, magnetisations_std, temps = Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps), Vector{Float64}(undef, T_Steps)
    @showprogress "Iterating over temperature..." for (iT, T) in enumerate(range(T_Start, T_End, T_Steps))
        lookup_table = create_lookup_table(T, J=J)
        grid = create_equilibrated_grid(grid_size=grid_size, J=J, lookup_table=lookup_table, T=T, B=B, N=N_Thermalize, initial_up_prob=initial_up_prob)

        energies_, magnetisations_ = sample_grid(grid, J, lookup_table, T=T, B=B, N=N_Sample, N_Subsweep=N_Subsweep)

        magnetisations[iT] = mean(abs.(magnetisations_))
        magnetisations_std[iT] = std(abs.(magnetisations_))
        # only take the absolute value of the magnetisation, because the system is symmetric
        energies[iT] = mean(energies_)
        energies_std[iT] = std(energies_)
        temps[iT] = T
    end
    return (energies, energies_std), (magnetisations, magnetisations_std), temps
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