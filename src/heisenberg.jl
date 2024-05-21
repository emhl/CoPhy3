using Statistics

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
        energy += -J * (x_grid[i, j, k] * (x_grid[mod1(i + 1, L), j, k] + x_grid[mod1(i - 1, L), j, k]))
        energy += -J * (y_grid[i, j, k] * (y_grid[i, mod1(j + 1, L), k] + y_grid[i, mod1(j - 1, L), k]))
        energy += -J * (z_grid[i, j, k] * (z_grid[i, j, mod1(k + 1, L)] + z_grid[i, j, mod1(k - 1, L)]))
    end
    return energy
end

function get_energy_diff(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}; pos::Tuple{Int64,Int64,Int64}, new_spin::Tuple{Float64,Float64,Float64}, J::Float64=1.0)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    i, j, k = pos
    energy_diff = 0.0
    energy_diff += -J * (new_spin[1] - x_grid[i, j, k]) * (x_grid[mod1(i + 1, L), j, k] + x_grid[mod1(i - 1, L), j, k])
    energy_diff += -J * (new_spin[2] - y_grid[i, j, k]) * (y_grid[i, mod1(j + 1, L), k] + y_grid[i, mod1(j - 1, L), k])
    energy_diff += -J * (new_spin[3] - z_grid[i, j, k]) * (z_grid[i, j, mod1(k + 1, L)] + z_grid[i, j, mod1(k - 1, L)])
    return energy_diff
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
    i, j, k = rand(1:L), rand(1:L), rand(1:L)
    new_spin = normalize_spin(rand([-1.0, 1.0], 3))
    energy_diff = get_energy_diff(grid, pos=(i, j, k), new_spin=new_spin, J=J)
    if energy_diff < 0 || rand() < get_state_prob(energy_diff, T)
        mag_diff = get_magnetization_diff(grid, pos=(i, j, k), new_spin=new_spin)
        x_grid[i, j, k], y_grid[i, j, k], z_grid[i, j, k] = new_spin
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

function wolff_cluster(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, pos::Tuple{Int64,Int64,Int64}, r::Tuple{Float64,Float64,Float64}, T::Float64, J::Float64)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    cluster = zeros(Bool, L, L, L)
    cluster[pos...] = true
    stack = [pos]
    while !isempty(stack)
        i, j, k = pop!(stack)
        for (di, dj, dk) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            i_, j_, k_ = mod1(i + di, L), mod1(j + dj, L), mod1(k + dk, L)
            if !cluster[i_, j_, k_] && rand() > get_bond_prob(r, (x_grid[i, j, k], y_grid[i, j, k], z_grid[i, j, k]), (x_grid[i_, j_, k_], y_grid[i_, j_, k_], z_grid[i_, j_, k_]), T, J)
                cluster[i_, j_, k_] = true
                push!(stack, (i_, j_, k_))
            end
        end
    end
    return cluster
end

function wolff_cluster_energy(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, cluster::Array{Bool,3}, J::Float64)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    energy = 0.0
    for c in findall(cluster)
        for (di, dj, dk) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            i, j, k = mod1(c[1] + di, L), mod1(c[2] + dj, L), mod1(c[3] + dk, L)
            energy += -J * (x_grid[c] * x_grid[i, j, k] + y_grid[c] * y_grid[i, j, k] + z_grid[c] * z_grid[i, j, k])
        end
    end
    return energy
end


function wolff_step(grid::Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}, T::Float64, J::Float64=1.0)
    x_grid, y_grid, z_grid = grid
    L = size(x_grid)[1]
    pos = rand(1:L), rand(1:L), rand(1:L)
    r = normalize_spin(rand([-1.0, 1.0], 3))

    cluster = wolff_cluster(grid, pos, r, T, J)

    energy_diff = -wolff_cluster_energy(grid, cluster, J)

    scalar_product = x_grid[cluster] .* r[1] + y_grid[cluster] .* r[2] + z_grid[cluster] .* r[3]

    mag_diff_x = -2 * scalar_product .* r[1]
    mag_diff_y = -2 * scalar_product .* r[2]
    mag_diff_z = -2 * scalar_product .* r[3]

    x_grid[cluster] += mag_diff_x
    y_grid[cluster] += mag_diff_y
    z_grid[cluster] += mag_diff_z

    mag_diff = [sum(mag_diff_x), sum(mag_diff_y), sum(mag_diff_z)]

    # normalize spins ?
    x_grid[cluster], y_grid[cluster], z_grid[cluster] = normalize_spins(x_grid[cluster], y_grid[cluster], z_grid[cluster])

    # spin_length = sqrt.(x_grid[cluster] .^ 2 + y_grid[cluster] .^ 2 + z_grid[cluster] .^ 2)
    # println(min(spin_length...), max(spin_length...))

    energy_diff += wolff_cluster_energy(grid, cluster, J)

    return (x_grid, y_grid, z_grid), energy_diff, mag_diff
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
        x_mag[i], y_mag[i], z_mag[i] = magnetisation
    end
    return energies, (x_mag, y_mag, z_mag)
end

function measure_single_config(; grid_size::Int=10, J::Float64=1.0, T::Float64=0.0, N_Sample::Int=1000, N_Thermalize::Int=100 * grid_size^3, N_Subsweep::Int=3 * grid_size^3, mc_algo::Function=metropolis_step)
    grid = thermalize_grid(grid_size=grid_size, J=J, T=T, N=N_Thermalize, mc_algo=mc_algo)
    energies, magnetisations = sample_grid(grid, T=T, J=J, N_Subsweep=N_Subsweep, N_Thermalize=N_Thermalize, N_Sample=N_Sample, mc_algo=mc_algo)
    return (mean(energies), std(energies)), ((mean(magnetisations[1]), mean(magnetisations[2]), mean(magnetisations[3])), (std(magnetisations[1]), std(magnetisations[2]), std(magnetisations[3])))
end


function temp_sweep(; grid_size::Int=10, J::Float64=1.0, T_min::Float64=0.1, T_max::Float64=5.0, T_Steps::Int=100, N_Sample::Int=1000, N_Thermalize::Int=1000, N_Subsweep::Int=1000, mc_algo::Function=metropolis_step)
    Ts = range(T_min, T_max, length=T_Steps)
    energies = Vector{Float64}(undef, length(Ts))
    energies_std = Vector{Float64}(undef, length(Ts))
    magnetisations = Array{Float64,4}(undef, length(Ts), 3)
    magnetisations_std = Array{Float64,4}(undef, length(Ts), 3)
    Threads.@threads for (i, T) in enumerate(Ts)
        (energies[i], energies_std[i]), (magnetisations[i], magnetisations_std[i]) = measure_single_config(grid_size=grid_size, J=J, T=T, N_Sample=N_Sample, N_Thermalize=N_Thermalize, N_Subsweep=N_Subsweep, mc_algo=mc_algo)
    end
    return Ts, (energies, energies_std), (magnetisations, magnetisations_std)
end

