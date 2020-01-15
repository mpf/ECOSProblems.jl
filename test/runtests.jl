using ECOSProblems
using LinearAlgebra
using Test

function dual_inf(x, z, bl, bu)
    dInf = 0.0
    jInf = 0
    for j = 1:n
	      blj = bl[j]
	      buj = bu[j]
	      if blj < buj
	          dj = z[j]
	          if dj > 0
		            dj =  dj * min(x[j] - blj, 1.0)
	          elseif dj < 0
		            dj = -dj * min(buj - x[j], 1.0)
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
    n = 2; m = 300
    A = randn(m, n)
    x0 = rand(n)
    b = A*x0
    u = rand(n)
    x, r, rNorm = lsbox(A, b, u, verbose=true)

    g = -A'r
    @test isapprox(norm(r), rNorm, atol=atol)
    @test isapprox(norm(b-A*x), norm(r), atol=atol)
    dInf, jInf = @enter dual_inf(x, g, zeros(n), u)
    println("dual Inf = $dInf, var = $jInf")
end

