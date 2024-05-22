using Statistics
using ProgressMeter


function normalize_spin(x::Float64, y::Float64, z::Float64)
    norm = sqrt(x^2 + y^2 + z^2)
    return x / norm, y / norm, z / norm
end

function normalize_spin(spin::Tuple{Float64,Float64,Float64})
    return normalize_spin(spin...)
end

function normalize_spin(spin::Array{Float64,1})
    return normalize_spin(spin...)
end


function normalize_spins(x::Array{Float64,1}, y::Array{Float64,1}, z::Array{Float64,1})
    norm = sqrt.(x .^ 2 + y .^ 2 + z .^ 2)
    return x ./ norm, y ./ norm, z ./ norm
end

function normalize_grid(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}})
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    for i in 1:L, j in 1:L, k in 1:L
        x_grid[i, j, k], y_grid[i, j, k], z_grid[i, j, k] = normalize_spin(x_grid[i, j, k], y_grid[i, j, k], z_grid[i, j, k])
    end
    return x_grid, y_grid, z_grid
end


function create_grid(L::Int64)
    # grid of 3D spins with random orientation
    grid = (rand([-1.0, 1.0], L, L, L), rand([-1.0, 1.0], L, L, L), rand([-1.0, 1.0], L, L, L))
    # normalize spins
    grid = normalize_grid(grid)
    return grid
end


function get_energy(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, J::Float64)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    energy = 0.0
    for i in 1:L, j in 1:L, k in 1:L
        neighbors = [(mod1(i + 1, L), j, k), (i, mod1(j + 1, L), k), (i, j, mod1(k + 1, L))]
        energy += -J * x_grid[i, j, k] * sum(x_grid[n...] for n in neighbors)
        energy += -J * y_grid[i, j, k] * sum(y_grid[n...] for n in neighbors)
        energy += -J * z_grid[i, j, k] * sum(z_grid[n...] for n in neighbors)
    end
    return energy
end

function get_energy_diff(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}; pos::Tuple{Int64,Int64,Int64}, new_spin::Tuple{Float64,Float64,Float64}, J::Float64=1.0)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    i, j, k = pos
    energy_diff = 0.0
    neighbors = [(mod1(i + 1, L), j, k), (mod1(i - 1, L), j, k), (i, mod1(j + 1, L), k), (i, mod1(j - 1, L), k), (i, j, mod1(k + 1, L)), (i, j, mod1(k - 1, L))]
    energy_diff += -J * (new_spin[1] - x_grid[i, j, k]) * (sum(x_grid[n...] for n in neighbors))
    energy_diff += -J * (new_spin[2] - y_grid[i, j, k]) * (sum(y_grid[n...] for n in neighbors))
    energy_diff += -J * (new_spin[3] - z_grid[i, j, k]) * (sum(z_grid[n...] for n in neighbors))
end

function get_magnetization(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}})
    x_grid, y_grid, z_grid = grid
    x_magnetization = sum(x_grid)
    y_magnetization = sum(y_grid)
    z_magnetization = sum(z_grid)
    return [x_magnetization, y_magnetization, z_magnetization]
end

function get_magnetization_diff(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}; pos::Tuple{Int64,Int64,Int64}, new_spin::Tuple{Float64,Float64,Float64})
    x_grid, y_grid, z_grid = grid
    i, j, k = pos
    x_magnetization_diff = new_spin[1] - x_grid[i, j, k]
    y_magnetization_diff = new_spin[2] - y_grid[i, j, k]
    z_magnetization_diff = new_spin[3] - z_grid[i, j, k]
    return [x_magnetization_diff, y_magnetization_diff, z_magnetization_diff]
end

function get_state_prob(energy_diff::Float64, T::Float64)
    return exp(-energy_diff / T)
end


function metropolis_step(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, T::Float64, J::Float64=1.0)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    pos = rand(1:L), rand(1:L), rand(1:L)
    new_spin = normalize_spin(rand([-1.0, 1.0], 3))
    energy_diff = get_energy_diff(grid, pos=pos, new_spin=new_spin, J=J)
    if rand() < get_state_prob(energy_diff, T)
        mag_diff = get_magnetization_diff(grid, pos=pos, new_spin=new_spin)
        x_grid[pos...], y_grid[pos...], z_grid[pos...] = new_spin
    else
        mag_diff = [0.0, 0.0, 0.0]
        energy_diff = 0.0
    end
    return (x_grid, y_grid, z_grid), energy_diff, mag_diff
end

function dot(x, y)
    return sum(x .* y)
end

function get_bond_prob(r::Tuple{Float64,Float64,Float64}, spin_i::Tuple{Float64,Float64,Float64}, spin_j::Tuple{Float64,Float64,Float64}, T::Float64, J::Float64)
    return 1 - exp(-2 * J / T * dot(r, spin_i) * dot(r, spin_j))
end


function nn_sum(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, pos::Tuple{Int64,Int64,Int64}; L::Int64=size(grid[1])[1])
    x_grid, y_grid, z_grid = grid
    i, j, k = pos
    neighbors = [(mod1(i + 1, L), j, k), (mod1(i - 1, L), j, k), (i, mod1(j + 1, L), k), (i, mod1(j - 1, L), k), (i, j, mod1(k + 1, L)), (i, j, mod1(k - 1, L))]
    return sum(x_grid[n...] for n in neighbors), sum(y_grid[n...] for n in neighbors), sum(z_grid[n...] for n in neighbors)
end


function wolff_flip(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, pos::Tuple{Int64,Int64,Int64}, r::Tuple{Float64,Float64,Float64}, dE::Float64=0.0, dM::Array{Float64,1}=[0.0, 0.0, 0.0]; T::Float64=1.0, J::Float64=1.0)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]

    spin = x_grid[pos...], y_grid[pos...], z_grid[pos...]
    new_spin = normalize_spin(spin .- 2 * dot(spin, r) .* r)
    dM_ = new_spin .- spin
    dM = dM .+ dM_
    dE -= J * dot(dM_, nn_sum(grid, pos, L=L))
    x_grid[pos...], y_grid[pos...], z_grid[pos...] = new_spin

    i, j, k = pos
    neighbors = [(mod1(i + 1, L), j, k), (mod1(i - 1, L), j, k), (i, mod1(j + 1, L), k), (i, mod1(j - 1, L), k), (i, j, mod1(k + 1, L)), (i, j, mod1(k - 1, L))]
    for n in neighbors
        if rand() < get_bond_prob(r, spin, (x_grid[n...], y_grid[n...], z_grid[n...]), T, J)
            grid, dE, dM = wolff_flip(grid, n, r, dE, dM, T=T, J=J)
        end
    end
    return grid, dE, dM
end


function wolff_step(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, T::Float64, J::Float64=1.0)
    L = size(grid[1])[1]
    pos = rand(1:L), rand(1:L), rand(1:L)
    r = normalize_spin(rand([-1.0, 1.0], 3))

    grid, dE, dM = wolff_flip(grid, pos, r, T=T, J=J)

    return grid, dE, dM
end


function thermalize_grid(; grid_size::Int=10, T::Float64=0.0, J::Float64=1.0, N::Int64=1000, mc_algo::Function=metropolis_step)
    grid = create_grid(grid_size)
    for i in 1:N
        grid, _ = mc_algo(grid, T, J)
    end
    return grid
end

function subsweep(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, T::Float64, J::Float64=1.0, N::Int64=1000; mc_algo::Function=metropolis_step)
    energy_diff, mag_diff = 0.0, [0.0, 0.0, 0.0]
    for i in 1:N
        grid, energy_diff_, mag_diff_ = mc_algo(grid, T, J)
        energy_diff += energy_diff_
        mag_diff += mag_diff_
    end
    return grid, energy_diff, mag_diff
end

function sample_grid(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}; T::Float64=0.0, J::Float64=1.0, N_Subsweep::Int64=1_000, N_Thermalize::Int64=1_000, N_Sample::Int64=1_000, mc_algo::Function=metropolis_step)
    energies = Vector{Float64}(undef, N_Sample)
    x_mag, y_mag, z_mag = Vector{Float64}(undef, N_Sample), Vector{Float64}(undef, N_Sample), Vector{Float64}(undef, N_Sample)
    energy = get_energy(grid, J)
    magnetisation = get_magnetization(grid)
    for i in 1:N_Sample
        grid, energy_diff, mag_diff = subsweep(grid, T, J, N_Subsweep, mc_algo=mc_algo)
        energy += energy_diff
        magnetisation += mag_diff
        energies[i] = energy
        x_mag[i], y_mag[i], z_mag[i] = abs.(magnetisation)
    end
    return energies, [x_mag, y_mag, z_mag]
end

function measure_single_config(; grid_size::Int=10, J::Float64=1.0, T::Float64=0.0, N_Sample::Int=1_000, N_Thermalize::Int=1000 * grid_size^3, N_Subsweep::Int=3 * grid_size^3, mc_algo::Function=metropolis_step)
    grid = thermalize_grid(grid_size=grid_size, J=J, T=T, N=N_Thermalize, mc_algo=mc_algo)
    energies, magnetisations = sample_grid(grid, T=T, J=J, N_Subsweep=N_Subsweep, N_Thermalize=N_Thermalize, N_Sample=N_Sample, mc_algo=mc_algo)
    binder_cumulant = 1 - mean(magnetisations[1] .^ 4 + magnetisations[2] .^ 4 + magnetisations[3] .^ 4) / (3 * mean(magnetisations[1] .^ 2 + magnetisations[2] .^ 2 + magnetisations[3] .^ 2)^2)
    return (mean(energies), std(energies)), (mean(mean(magnetisations)), mean(std(magnetisations))), binder_cumulant
end


function temp_sweep(; grid_size::Int=10, J::Float64=1.0, T_min::Float64=0.1, T_max::Float64=5.0, T_Steps::Int=100, N_Sample::Int=10_000, N_Thermalize::Int=100 * grid_size^3, N_Subsweep::Int=3 * grid_size^3, mc_algo::Function=metropolis_step)
    Ts = range(T_min, T_max, length=T_Steps)
    energies = Vector{Float64}(undef, length(Ts))
    energies_std = Vector{Float64}(undef, length(Ts))
    magnetisations = Array{Float64}(undef, length(Ts))
    magnetisations_std = Array{Float64}(undef, length(Ts))
    binder_cum = Vector{Float64}(undef, length(Ts))
    @showprogress Threads.@threads for (i, T) in collect(enumerate(Ts))
        (energies[i], energies_std[i]), (magnetisations[i], magnetisations_std[i]), binder_cum[i] = measure_single_config(grid_size=grid_size, J=J, T=T, N_Sample=N_Sample, N_Thermalize=N_Thermalize, N_Subsweep=N_Subsweep, mc_algo=mc_algo)
    end
    return Ts, (energies, energies_std), (magnetisations, magnetisations_std), binder_cum
end

