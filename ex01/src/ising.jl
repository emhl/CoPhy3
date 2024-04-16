function create_grid(N1::Int, N2::Int, N3::Int)
    return rand([-1,1],N1,N2,N3)
end

function energy(grid::Array{Int, 3},J::Float64, B::Float64=0.0)
    N1, N2, N3 = size(grid)
    E = 0.0
    for i in 1:N1
        for j in 1:N2
            for k in 1:N3
                E += -J*grid[i,j,k]*(grid[mod1(i+1,N1),j,k] + grid[i,mod1(j+1,N2),k] + grid[i,j,mod1(k+1,N3)])
            end
        end
    end
    return E
end

function magnetisation(grid::Array{Int, 3})
    return sum(grid)/length(grid)
end


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
        grid=copy(grid_tmp)
    end
    return grid
end