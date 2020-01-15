using ECOSProblems
using LinearAlgebra
using Test

function dual_inf(x, z, bl, bu)
    dInf = 0.0; jInf = 0
    for j = 1:n
	      if bl[j] < bu[j]
	          dj = z[j]
	          if dj > 0
		            dj =  dj * min(x[j] - bl[j], 1.0)
	          elseif dj < 0
		            dj = -dj * min(bu[j] - x[j], 1.0)
            end
	          if dInf < dj
		            dInf = dj
		            jInf = j
            end
	      end
    end
    return dInf, jInf
end

@testset "ECOSProblems.jl" begin
    atol = 1e-6
    n = 3; m = 100
    A = randn(m, n)
    x0 = rand(n)
    b = A*x0
    u = rand(n)
    x, r, rNorm = lsbox(A, b, u, verbose=false)
    @test isapprox(norm(r), rNorm, atol=atol)
    @test isapprox(norm(b-A*x), norm(r), atol=atol)
    dInf, jInf = dual_inf(x, -(A'r), zeros(n), u)
    @test dInf ≤ 1e-3
    # println("dual Inf = $dInf, var = $jInf, $(g[jInf])   $(u[jInf]-x[jInf]) ")
end

