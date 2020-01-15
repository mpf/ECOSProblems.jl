mutable struct ECOSls <: ECOSWrapper
    ecos_ptr::Ptr{ECOS.Cpwork}
    A1::AbstractMatrix{Float64}
    b1::Vector{Float64}
    A2::AbstractMatrix{Float64}
    b2::Vector{Float64}
    n_ecos::Int
    n::Int
    m1::Int
    m2::Int
end

mutable struct ECOSlssplx <: ECOSWrapper
    ecos_ptr::Ptr{ECOS.Cpwork}
    A::AbstractMatrix{Float64}
    b::Vector{Float64}
    n_ecos::Int
    n::Int
    m::Int
end

"""
Solve the constrained least-squares problem

    minimize  ||A1x-b1||₂ st A2x=b2, x≥0.

Rephrases the problem as linear SOC program:

    minimize  [0]'[x]  st  [A1 0  I][x]    [b1] m1
              [1] [t]      [A2 0  0][t] =  [b2] m2
              [0] [r]               [r]
                            n  1  m1
                               x, (t,r) ∈  Rn₊ x (Qn₊)

with dimensions  len(x) = n   len(b1) = m1
                 len(r) = m1  len(b2) = m2

Solve using ECOS, which formulates linear SOC programs as

    minimize  c'x  st  A*x=b, G*x + s = h

Thus, set the ECOS variables as

  c      = [0 , 1,  0]
  A      = [A1  0   I]
            A2  0   0]
  G      = -Identity(n+1+m1)
  h      = [0   0   0]
"""
function ECOSls(A1::AbstractMatrix, b1::Vector{Float64},
                A2::AbstractMatrix, b2::Vector{Float64};
                kwargs...)
    m1, n1 = size(A1)
    m2, n2 = size(A2)
    n1 != n2 && throw(DimensionMismatch("matrix sizes not compatible"))
    n = n1

    n_ecos = n + 1 + m1  # number of vars == len(x)
    m = n_ecos           # number of inequality constraints == len(h)
    p = m1 + m2          # number of equality constraints == len(b)
    l = n                # number of vars in positive orthant
    ncones = 1           # number of SOC cones
    q = [1+m1]           # number of vars in the SOC
    e = 0                # number of exponential cones
    G = ECOS.ECOSMatrix(sparse(-1.0I,n_ecos,n_ecos))
    A = ECOS.ECOSMatrix([A1  spzeros(m1,1)  sparse(1.0I,m1,m1)
                         A2  spzeros(m2,1)      spzeros(m2,m1)])
    c = zeros(n_ecos); c[n+1] = 1.0
    h = zeros(n_ecos)
    b = vcat(b1, b2)

    ecos_ptr = ECOS.setup(n_ecos, m, p, l, ncones, q, e, G, A, c, h, b; kwargs...)
    # problem = ECOSls(ecos_ptr, A1, b1, A2, b2, n_ecos, n, m1, m2)
    ECOS.solve(ecos_ptr)
    # finalizer(cleanup!, problem)
    # return ecos_ptr
end

"""
    ls_simplex(A, b)

Setup the constrained least-squares problem

    minimize  ||Ax-b||₂  subj to  sum(x)=1, x≥0
"""
function ls_simplex(A::AbstractMatrix, b::Vector{Float64}; kwargs...)
    m, n = size(A)
    lsg = ECOSls(A, b, ones(1,n), [1.0]; kwargs...)
    return ECOSlssplx(lsg.ecos_ptr, A, b, lsg.n_ecos, n, m)
end

function solve!(p::ECOSlssplx)

    exitflag = ECOS.solve(p.ecos_ptr)

    if exitflag != ECOS.ECOS_OPTIMAL
        error("did not solve to optimality --- $exitflag")
    end

    # Extract the primal and dual solutions of the ECOS problem
    ecos_prob = unsafe_wrap(Array, p.ecos_ptr, 1)[1]
    xtr = unsafe_wrap(Array, ecos_prob.x, p.n_ecos)[:]
    du_sol_eq = unsafe_wrap(Array, ecos_prob.y, p.m)[:]
    du_sol_ineq = unsafe_wrap(Array, ecos_prob.z, p.n_ecos)[:]
    ECOS.cleanup(p.ecos_ptr)
    # Extract intended solution
    # x = xtr[1:p.n]
    # r = xtr[p.n+2:end]
    # z = du_sol_ineq[1:p.n]
    # μ = du_sol_ineq[p.n+1]

    return xtr
end
