using ProgressMeter
import StatsBase as sb

function init_system(; N_particles::Int64=10, L::Float64=10.0, v0::Float64=1.0)
    # random positions from 0 to L in 3 Dimensions
    return L * rand(N_particles,3), 2* v0 * rand(N_particles,3) .- v0
end

function calc_distances(pos_i::AbstractVector, pos_j::AbstractVector ;cutoff::Float64=1.0, L::Float64=10.0)
    # ToDo: add periodic boundary conditions
    dist = pos_i .- pos_j
    for (i, d) in enumerate(dist)
        if abs(d) > L/2
            dist[i] = L - abs(d)
        end
    end
    return dist
end

function calc_lj_force(pos_i::AbstractVector, pos_j::AbstractVector; cutoff::Float64=1.0, L::Float64=10.0)
    dist = calc_distances(pos_i, pos_j, cutoff=cutoff,L=L)

    r_dist = sb.norm(dist)

    # calculate the force from the Lennard-Jones potential
    if r_dist > cutoff
        return zeros(length(dist))
    else
        return -24 .* (2 .* r_dist .^ -14 .- r_dist .^ -8) .* dist
    end

end


function verlet_step(pos::AbstractVector, pos_last::AbstractArray; force::AbstractVector, dt::Float64=0.01)
    # calculate the new position
    return 2.0.*pos .- pos_last .+ dt^2 .* force
end

function verlet_simulate(; N_particles::Int64=10, L::Float64=10.0, dt::Float64=0.01, steps::Int64=100, cutoff::Float64=2.5)
    positions, velocities = init_system(N_particles=N_particles, L=L)
    positions_last = positions .- velocities .* dt
    
    positions_arr = zeros(steps+1, N_particles, 3)
    positions_arr[1,:,:] = positions

    velocities_arr = zeros(steps+1, N_particles, 3)
    velocities_arr[1,:,:] = velocities
    @showprogress for s in 1:steps
        positions_next = zeros(N_particles, 3)

        forces = zeros(N_particles,3)
        Threads.@threads for i in 1:N_particles, j in 1:N_particles
            if i != j
                forces[i,:] .+= calc_lj_force(positions[i,:], positions[j,:], cutoff=cutoff,L=L)
            end
        end

        positions_next = verlet_step(positions, positions_last, force=forces, dt=dt)
        
        velocities = (positions_next .- positions_last) ./ (2*dt)

        positions_last = positions
        positions = positions_next

        positions_arr[s+1,:,:] = positions
        velocities_arr[s+1,:,:] = velocities
    end
    return positions_arr, velocities_arr
end


function calc_lj_potential(pos_i::AbstractArray, pos_j::AbstractArray; cutoff::Float64=1.0, L::Float64=10.0)
    dist = calc_distances(pos_i, pos_j, cutoff=cutoff,L=L)
    r_dist = sb.norm(dist)
    return 4.0 * sum(r_dist .^ -12 .- r_dist .^ -6)
end


function calc_lj_hamiltonian(positions::Array{Float64,3}, velocities::Array{Float64,3}; cutoff::Float64=1.0,L::Float64=10.0)
    kinetic_energy = sum(velocities .^ 2 , dims=[2,3]) 
    steps = length(kinetic_energy)
    kinetic_energy = reshape(kinetic_energy, steps)
    # println(steps)
    N_particles = size(positions,2)
    potential_energy = zeros(steps)

    cutoff_potential = 4*(1/cutoff^12 - 1/cutoff^6)
    
    Threads.@threads for s in 1:steps
        for i in 1:N_particles, j in 1:N_particles
            if i != j
                potential_energy[s] += calc_lj_potential(positions[s+1,i,:], positions[s+1,j,:], cutoff=cutoff, L=L)
            end
        end
    end


    return 0.5 .* (kinetic_energy .+ potential_energy .- cutoff_potential)
end

