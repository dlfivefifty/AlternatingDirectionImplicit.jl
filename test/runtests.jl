using AlternatingDirectionImplicit, ClassicalOrthogonalPolynomials, MatrixFactorizations, Test
using ClassicalOrthogonalPolynomials: plan_grid_transform

@testset "Fortunato–Townsend" begin
    P = Legendre()
    Q = Weighted(Jacobi(1,1))
    n = 100
    kr = Base.OneTo(n)
    Δ = (diff(Q)'diff(Q))[kr,kr]
    M = (Q'Q)[kr,kr]
    QP = (Q'P)[kr,kr]

    f = (x,y) -> -2 *sin(π*x) * (2π*y *cos(π*y) + (1-π^2*y^2) *sin(π*y))
    (x,y),pl =  plan_grid_transform(P, (n,n))
    F = QP*(pl*f.(x, y'))*QP'
    
    X = adi(F, Mₙ, -Mₙ, Δₙ; factorize = reversecholesky ∘ Symmetric)
    @test Mₙ*X*Δₙ + Δₙ*X*Mₙ ≈ F

    u_exact = (x,y) -> sin(π*x)*sin(π*y)*y^2
    @test Q[0.1,1:n]'*X*Q[0.2,1:n] ≈ u_exact(0.1,0.2)
end