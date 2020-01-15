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
