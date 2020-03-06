"""
Use ECOS to solve the bound-constrained least-squares problem

    minimize ||Ax-b||_2 subj to ℓ ≤ x ≤ u.

ECOS is a primal-dual interior method for SOCP. This module translates
the problem into the SOCP

    minimize  t  subj to Ax + r = b, (||x||,t) ∈ SOC, ℓ ≤ x ≤ u.

See [`lsbox`](@ref)
"""
module ECOSProblems

using ECOS
using LinearAlgebra
using SparseArrays

# export ls_setup, ls_simplex, solve!, cleanup!
export lsbox
# export ECOSls

include("lsbox.jl")

# abstract type ECOSWrapper end

# solve!(p::ECOSWrapper) = ECOS.solve(p.ecos_ptr)
# cleanup!(p::ECOSWrapper) = ECOS.cleanup(p.ecos_ptr)

end # module
