function create_grid(N1::Int, N2::Int, N3::Int)
    return rand([-1,1],N1,N2,N3)
end







@doc "Energy calculation for different input arguments"
function energy(grid::Array{Int, 3},J::Float64, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    E = 0.0
    for i in 1:N1
        for j in 1:N2
            for k in 1:N3
                E += -J*grid[i,j,k]*(
                        grid[mod1(i+1,N1),j,k] 
                        + grid[i,mod1(j+1,N2),k] 
                        + grid[i,j,mod1(k+1,N3)]
                    )
            end
        end
    end
    E -= B*sum(grid)
    return E
end

@doc "Magnetisation of a given grid"
function magnetisation(grid::Array{Int, 3})
    return sum(grid)/length(grid)
end








@doc "Metropolitan Step function for different input arguments"
function metropolis_step(grid::Array{Int, 3}, J::Float64, T::Float64=0.0, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    i = rand(1:N1)
    j = rand(1:N2)
    k = rand(1:N3)
    E = energy(grid,J,B)
    grid_tmp = copy(grid)
    grid_tmp[i,j,k] *= -1
    E_tmp = energy(grid_tmp,J,B)
    if rand() < exp(-(E_tmp-E)/T)
        grid=grid_tmp
    end
    return grid
end

@doc "Function for n repetitions of metropolis_step, tracking the energy and other stats, maybe grid depends on size"
function MonteCarloConstantTemp(grid::Array{Int,3}, J::Float64, T::Float64, B::Float64, n::Int64)

    energies, magnetisations = Float64[], Float64[];

        for i in 1:n
            grid = metropolis_step(grid, J, T, B);
            push!(energies, energy(grid,J,B));
            push!(magnetisations, magnetisation(grid));
        end
    
    return grid, energies, magnetisations;
    
end


@doc "function for sweeping over a field intervall using B_Steps steps"
function field_sweep(grid_dimension::Int=10, J::Float64=1.0, T::Float64=0.0, B_Start::Float64=0.0, B_End::Float64=1.0,B_Steps::Int=100,n::Int=100_000)
    energies, magnetisations, field = Float64[], Float64[], Float64[]
    grid=create_grid(grid_dimension,grid_dimension,grid_dimension)

    for B in range(B_Start,B_End,B_Steps)
            
        for i in 1:n
            grid = metropolis_step(grid, J, T, B);
        end
        push!(magnetisations, magnetisation(grid));
        push!(energies, energy(grid,J,B));
        push!(field, B)
    end
    return energies, magnetisations, field
end

@doc "function for sweeping over a temperature intervall using T_Steps steps"
function temp_sweep(grid_dimension::Int=10, J::Float64=1.0, T_Start::Float64=0.0, T_End::Float64=10.0, B::Float64=0.0, T_Steps::Int=100,n::Int=100_000)
    energies, magnetisations, temp = Float64[], Float64[], Float64[]
    grid=create_grid(grid_dimension,grid_dimension,grid_dimension)

    for T in range(T_Start,T_End,T_Steps)
            
        for i in 1:n
            grid = metropolis_step(grid, J, T, B);
        end
        push!(magnetisations, magnetisation(grid));
        push!(energies, energy(grid,J,B));
        push!(temp, T)
    end
    return energies, magnetisations, temp
end


