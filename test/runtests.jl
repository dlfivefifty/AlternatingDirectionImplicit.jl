using AlternatingDirectionImplicit, ClassicalOrthogonalPolynomials, MatrixFactorizations, Test
using ClassicalOrthogonalPolynomials: plan_grid_transform

@testset "standard ADI" begin
    m,n = 5,6
    A = Symmetric(randn(m,m))+ 6I
    B = -(Symmetric(randn(n,n)) + 6I)

    a,b = extrema(eigvals(Symmetric(A)))
    c,d = extrema(eigvals(Symmetric(B)))
    F = randn(m,n)
    L = kron(I(n),A) - kron(B,I(m))
    X_ex = reshape(L \ vec(F), m,n)
    @test A*X_ex - X_ex*B ≈ F

    for tol = 10.0 .^ (-(1:14))
        γ = abs(c-a)*abs(d-b)/(abs(c-b)*abs(d-a))
        J = ceil(Int, log(16γ)*log(4/tol)/π^2)
        p,q = AlternatingDirectionImplicit.adi_shifts(J, a, b, c, d, tolerance)

        X = zeros(m,n)
        for j = 1:J
            X = (F - (A - p[j]*I)*X) / (B - p[j]*I)
            X = (A - q[j]*I) \ (F - X*(B - q[j]*I))
        end

        @test norm(X - X_ex) ≤ tol*norm(X_ex)
    end
end

@testset "Generalised ADI" begin
    m,n = 5,6
    A = Symmetric(randn(m,m))+ 6I
    B = -(Symmetric(randn(n,n)) + 6I)
    C = Symmetric(randn(n,n))+ 6I
    D = Symmetric(randn(m,m))+ 6I

    a,b = extrema(eigvals(Symmetric(A),Symmetric(D)))
    c,d = extrema(eigvals(Symmetric(B),Symmetric(C)))
    F = randn(m,n)
    L = kron(C,A) - kron(B,D)
    X = reshape(L \ vec(F), m,n)
    @test A*X*C - D*X*B ≈ F

    L = reversecholesky(C).L
    V = reversecholesky(D).L

    Y = V*X*L'
    @test A*inv(V)*Y*L - V'*Y*inv(L')*B ≈ F
    
    Ã = inv(V')*A*inv(V)
    B̃ = inv(L')*B*inv(L)
    G = V'\F/L
    
    @test Ã*Y - Y*B̃ ≈ G

    @test all(extrema(eigvals(Ã)) .≈ (a,b))
    @test all(extrema(eigvals(B̃)) .≈ (c,d))

    for tol = 10.0 .^ (-(1:14))
        γ = abs(c-a)*abs(d-b)/(abs(c-b)*abs(d-a))
        J = ceil(Int, log(16γ)*log(4/tol)/π^2)
        p,q = AlternatingDirectionImplicit.adi_shifts(J, a, b, c, d, tolerance)

        Yⱼ = zeros(m,n)
        for j = 1:J
            Yⱼ = (G - (Ã - p[j]*I)*Yⱼ) / (B̃ - p[j]*I)
            Yⱼ = (Ã - q[j]*I) \ (G - Yⱼ*(B̃ - q[j]*I))
        end

        @test norm(Yⱼ - Y) ≤ tol*norm(Y)
    end

    for tol = 10.0 .^ (-(1:14))
        γ = abs(c-a)*abs(d-b)/(abs(c-b)*abs(d-a))
        J = ceil(Int, log(16γ)*log(4/tol)/π^2)
        p,q = AlternatingDirectionImplicit.adi_shifts(J, a, b, c, d, tolerance)

        Yⱼ = zeros(m,n)
        j = 1
        Ỹⱼ = (G - (Ã - p[j]*I)*Yⱼ) / (B̃ - p[j]*I)
        @test Ỹⱼ ≈ inv(V')*(F - (A - p[j]*D)*inv(V)*Yⱼ*L) * inv(B - p[j]*C)*L'

        Zⱼ = (Ã - q[j]*I) \ (G - Ỹⱼ*(B̃ - q[j]*I))
        @test Zⱼ ≈ V * inv(A - q[j]*D) * (F - V'*Ỹⱼ*inv(L')*(B - q[j]*C)) *inv(L)
    
        Zⱼ = zeros(m,n)
        for j = 1:J
            Zⱼ = (F - (A - p[j]*D)*Zⱼ)/(B - p[j]*C)
            Zⱼ = (A - q[j]*D)\(F - Zⱼ*(B - q[j]*C))
        end

        @test norm(V*Zⱼ*inv(L) - Y) ≤ tol*norm(Y)
    end
end

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