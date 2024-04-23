function create_grid(N1::Int, N2::Int, N3::Int)
    return rand([-1, 1], N1, N2, N3)
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

# create function that only calculates the anergy diference that the single flip would cause
function energy_diff(grid::Array{Int,3}, flip_position::Tuple{Int,Int,Int}; J::Float64=1.0, B::Float64=0.0 )
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


@doc "Magnetisation of a given grid"
function magnetisation(grid::Array{Int,3})
    return sum(grid) / length(grid)
end


function state_probability(E::Float64, T::Float64)
    return exp(-E / T)
end

function mean_observable(values::Array{Float64,1}, weights::Array{Float64,1})
    #TODO can this function be vectorized?
    # return sum(values .* weights) / sum(weights)
    s = 0.0
    for (value,weight) in zip(values, weights)
        s += value * weight
    end
    return s / sum(weights) # normalize
end

function mean_observable_squared(values::Array{Float64,1}, weights::Array{Float64,1})
    #TODO can this function be vectorized?
    # return sum((values .^ 2) .* weights) / sum(weights)
    s = 0.0
    for (value,weight) in zip(values, weights)
        s += value^2 * weight
    end
    return s / sum(weights) # normalize
end

function magnetic_suceptibility(magnetisation_values::Array{Float64,1}, weights::Array{Float64,1})
    return mean_observable_squared(magnetisation_values, weights) - mean_observable(magnetisation_values, weights)^2
end


function heat_capacity(energy_values::Array{Float64,1}, weights::Array{Float64,1}, T::Float64)
    return (mean_observable_squared(energy_values, weights) - mean_observable(energy_values, weights)^2) / T^2
end


@doc "Metropolitan Step function for different input arguments"
function metropolis_step(grid::Array{Int,3}, J::Float64, T::Float64=0.0, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    i = rand(1:N1)
    j = rand(1:N2)
    k = rand(1:N3)
    dE = energy_diff(grid, (i, j, k), J=J, B=B)
    if rand() < exp(-dE / T)
        grid[i, j, k] *= -1
    end
    return grid
end



@doc "Function for n repetitions of metropolis_step, tracking the energy and magnetization for constant temperature and field"
function monte_carlo_const_temp(grid::Array{Int,3}, J::Float64, T::Float64, B::Float64, n::Int64)

    energies, magnetisations = Float64[], Float64[]

    for i in 1:n
        grid = metropolis_step(grid, J, T, B)
        push!(energies, energy(grid, J, B))
        push!(magnetisations, magnetisation(grid))
    end

    return grid, energies, magnetisations

end


@doc "function for sweeping over a field intervall using B_Steps steps"
function field_sweep(;grid_size::Int=10, J::Float64=1.0, T::Float64=0.0, B_Start::Float64=0.0, B_End::Float64=1.0, B_Steps::Int=100, N::Int=100_000)
    energies, magnetisations, field = Float64[], Float64[], Float64[]
    for B in range(B_Start, B_End, B_Steps)
        grid = create_grid(grid_size, grid_size, grid_size) # always start with a new random grid
        for i in 1:N
            grid = metropolis_step(grid, J, T, B)
        end
        push!(magnetisations, magnetisation(grid))
        push!(energies, energy(grid, J, B))
        push!(field, B)
    end
    return energies, magnetisations, field
end


@doc "function for sweeping over a temperature intervall using T_Steps steps"
function temp_sweep(;grid_size::Int=10, J::Float64=1.0, T_Start::Float64=0.0, T_End::Float64=10.0, B::Float64=0.0, T_Steps::Int=100, N::Int=100_000)
    energies, magnetisations, temp = Float64[], Float64[], Float64[]

    for T in range(T_Start, T_End, T_Steps)
        grid = create_grid(grid_size, grid_size, grid_size) # always start with a new random grid
        for i in 1:N
            grid = metropolis_step(grid, J, T, B)
        end
        push!(magnetisations, magnetisation(grid))
        push!(energies, energy(grid, J, B))
        push!(temp, T)
    end
    return energies, magnetisations, temp
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