"""
Solve the constrained least-squares problem

    minimize  ||Ax-b||₂  st  0 ≤ x ≤ u.

Rephrases the problem as linear SOC program:

    minimize  [0]'[x]  st  [A 0 I][x] = [b] m
              [0] [r]             [t]
                                  [r]
                            n 1 m

                               x, (t,r) ∈  Rn₊ x (Qn₊)

with dimensions  len(x) = n  len(b) = m
                 len(r) = m

Solve using ECOS, which formulates linear SOC programs as

    minimize  c'x  st  A*x=b, G*x + s = h, s in cone
"""
function lsbox(Ain::Matrix{Float64}, b::Vector{Float64},
               u::Vector{Float64}=zeros(0); kwargs...)
    m, n = size(Ain)
    nu = length(u)       # number of vars x with upper bound
    n_ecos = n + 1 + m   # number of vars == len(x)
    p = m                # number of equality constraints == len(b)
    l = n + nu           # number of vars in positive orthant
    ncones = 1           # number of SOC cones
    q = [1+m]            # number of vars in the SOC
    e = 0                # number of exponential cones

    GG = [ sparse(-1.0I,n ,n)  spzeros(n ,1)       spzeros(n ,m)
           sparse(+1.0I,nu,n)  spzeros(nu,1)       spzeros(nu,m)
                spzeros(1 ,n)      -1.0            spzeros( 1,m)
                spzeros(m ,n)  spzeros(m ,1)  sparse(-1.0I,m ,m)]
    G = ECOS.ECOSMatrix(GG)
    h = vcat(zeros(n), u,   0,  zeros(m))
    m_ecos =       n + nu + 1 + m
    A = ECOS.ECOSMatrix([Ain spzeros(m,1) sparse(1.0I,m,m)])
    c = zeros(n_ecos); c[n+1] = 1.0

    ecos_ptr = ECOS.setup(n_ecos, m_ecos, p, l, ncones, q, e,
                          G, A, c, h, b; kwargs...)
    ECOS.solve(ecos_ptr)
    # finalizer(cleanup!, problem)

    # Extract the primal and dual solutions of the ECOS problem
    ecos_prob = unsafe_wrap(Array, ecos_ptr, 1)[1]
    xtr = unsafe_wrap(Array, ecos_prob.x, n_ecos)[:]

    ECOS.cleanup(ecos_ptr, 0)
    x = xtr[1:n]
    t = xtr[n+1] # t = norm(r)
    r = xtr[n+2:end]
    return x, r, t
end
