module AlternatingDirectionImplicit

using HypergeometricFunctions, Elliptic, LinearAlgebra

export ADI

# These 4 routines from ADI were lifted from Kars' M4R repo.
function mobius(z, a, b, c, d, α)
    t₁ = a*(-α*b + b + α*c + c) - 2b*c
    t₂ = a*(α*(b+c) - b + c) - 2α*b*c
    t₃ = 2a - (α+1)*b + (α-1)*c
    t₄ = -α*(-2a+b+c) - b + c

    (t₁*z + t₂)/(t₃*z + t₄)
end

ellipticK(z) = convert(eltype(α),π)/2*HypergeometricFunctions._₂F₁(one(α)/2,one(α)/2,1, z)


function ADI_shifts(J, a, b, c, d, tol=1e-15)
    γ = (c-a)*(d-b)/((c-b)*(d-a))
    α = -1 + 2γ + 2√Complex(γ^2 - γ)
    α = Real(α)

    # K = ellipticK(1-1/big(α)^2)
    if α < 1e7
        K = Elliptic.K(1-1/α^2)
        dn = Elliptic.Jacobi.dn.((2*(0:J-1) .+ 1)*K/(2J), 1-1/α^2)
    else
        K = 2log(2)+log(α) + (-1+2log(2)+log(α))/α^2/4
        m1 = 1/α^2
        u = (1/2:J-1/2) * K/J
        dn = @. sech(u) + m1/4 * (sinh(u)cosh(u) + u) * tanh(u) * sech(u)
    end

    [mobius(-α*i, a, b, c, d, α) for i = dn], [mobius(α*i, a, b, c, d, α) for i = dn]
end

function ADI(A, B, C, F, a, b, c, d; tolerance=1e-15, factorize=factorize)
    "ADI method for solving standard sylvester AX - XB = F"
    # Modified slightly by John to allow for the mass matrix
    n = size(A)[1]
    X = zeros(axes(A))

    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tolerance)/π^2))
    # J = 200
    p, q = ADI_shifts(J, a, b, c, d, tolerance)

    for j = 1:J
        X = ((A/p[j] - C)*X - F/p[j])/factorize(C - B/p[j])
        X = factorize(C - A/q[j])\(X*(B/q[j] - C) - F/q[j])
    end

    X
end

end # module AlternatingDirectionImplicit
