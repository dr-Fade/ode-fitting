module OdeFitting

using DynamicalSystems, LinearAlgebra, DifferentialEquations

function fit_ode_to_time_series(g::Dataset, ▵t)
    # check that the data passed has acceptable dimensions
    m = length(g) - 1
    d = dimension(g)
    if d ≠ 3 || m < 20
        error("Only 3d model with more than 20 points is accepted!")
    end
    # construct a matrix of numerical derivatives
    D = hcat((g[begin+1:end].data - g[begin:end-1].data)'...) / ▵t
    g = vcat(g[begin+1:end].data'...)
    # allocate space for the jacobian matrix
    J = zeros(Float32, 3*m, 20)
    # linear components
    for i in 0:2
        J[i*m+1:(i+1)*m, i*(d+1)+1] = ones(Float32, m)
        J[i*m+1:(i+1)*m, i*(d+1)+2:(i+1)*(d+1)] = g
    end
    # nonlinear components
    # col 13
    J[1:m, 13] = g[:,1] .* g[:,2]
    J[m+1:2m, 13] = -g[:,1] .^ 2
    # col 14
    J[1:m, 14] = g[:,2] .^ 2
    J[m+1:2m, 14] = -g[:,1] .* g[:,2]
    # col 15
    J[1:m, 15] = g[:,1] .* g[:,3]
    J[2m+1:3m, 15] = -g[:,1] .^ 2
    # col 16
    J[1:m, 16] = g[:,2] .* g[:,3]
    J[2m+1:3m, 16] = -g[:,1] .* g[:,2]
    # col 17
    J[1:m, 17] = g[:,3] .^ 2
    J[2m+1:3m, 17] = -g[:,1] .* g[:,3]
    # col 18
    J[m+1:2m, 18] = g[:,1] .* g[:,3]
    J[2m+1:3m, 18] = -g[:,1] .* g[:,2]
    # col 19
    J[m+1:2m, 19] = g[:,2] .* g[:,3]
    J[2m+1:3m, 19] = -g[:,2] .^ 2
    # col 20
    J[m+1:2m, 20] = g[:,3] .^ 2
    J[2m+1:3m, 20] = -g[:,2] .* g[:,3]
    #check rank and assign μ
    μ = if rank(J) < 20 rand() else 0 end
    Y = (J' * J + μ * I(20)) ^ -1 * J' * D'
    println("Y = $Y")
    println("Proceeding to calculate tre trajectory...")

    u0 = [-10.,-6.,0]
    tspan = (0.00,25.0)
    prob = ODEProblem(model, u0, tspan, Y)
    sol = solve(prob)
    return sol.u |> Dataset, Y
end

function model(du,u,p,t)
    x, y, z = u
    c1, a11, a12, a13,
    c2, a21, a22, a23,
    c3, a31, a32, a33,
    p, s, q, d, h, e, f, r = p
    du[1] = c1 + a11*x + a12*y + a13*z + p*x*y + s*y^2 + q*x*z + d*y*z + h * z^2
    du[2] = c2 + a21*x + a22*y + a23*z - p*x^2 - s*x*y + e*x*z + f*y*z + r*z^2
    du[3] = c3 + a31*x + a32*y + a33*z - q*x^2 - (d+e)*x*y - f*y^2 - h*x*z - r*y*z
end

function fit_lorenz()
    T = 25
    dt = 0.01
    u0 = [12.5, 2.5, 1.5]
    lorenz = DynamicalSystems.Systems.lorenz(u0; σ = 10.0, ρ = 28.0, β = 8/3)
    g = DynamicalSystems.trajectory(lorenz, T; dt=dt)[1000:end,:][1:500]
    return fit_ode_to_time_series(g, dt)
end

end # module
