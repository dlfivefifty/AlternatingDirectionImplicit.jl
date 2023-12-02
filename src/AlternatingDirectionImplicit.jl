module AlternatingDirectionImplicit

using HypergeometricFunctions, Elliptic, LinearAlgebra
import Base: *
export adi, plan_adi!

# These 4 routines from ADI were lifted from Kars' M4R repo.
function mobius(z, a, b, c, d, α)
    t₁ = a*(-α*b + b + α*c + c) - 2b*c
    t₂ = a*(α*(b+c) - b + c) - 2α*b*c
    t₃ = 2a - (α+1)*b + (α-1)*c
    t₄ = -α*(-2a+b+c) - b + c

    (t₁*z + t₂)/(t₃*z + t₄)
end

# elliptick(z) = convert(eltype(α),π)/2*HypergeometricFunctions._₂F₁(one(α)/2,one(α)/2,1, z)


function adi_shifts(J, a, b, c, d, tol=1e-15)
    γ = (c-a)*(d-b)/((c-b)*(d-a))
    α = -1 + 2γ + 2√Complex(γ^2 - γ)
    α = Real(α)

    # K = elliptick(1-1/big(α)^2)
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
"ADI method for solving standard sylvester AX - XB = F"

struct ADIPlan{T, AA, BB, CC, DD}
    As::AA
    Bs::BB
    Cfacs::CC
    Dfacs::DD
    p::Vector{T}
    q::Vector{T}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
end




function plan_adi!(A, B, C, a, b, c, d; tolerance=1e-15, factorize=factorize)
    γ = (c-a)*(d-b)/((c-b)*(d-a))
    J = Int(ceil(log(16γ)*log(4/tolerance)/π^2))
    # J = 200
    p, q = adi_shifts(J, a, b, c, d, tolerance)
    ADIPlan([(A/p[j] - C) for j = 1:J], 
            [(B/q[j] - C) for j=1:J],
            [factorize(C - B/p[j]) for j=1:J],
            [factorize(C - A/q[j]) for j=1:J],
            p, q, 
            Matrix{eltype(A)}(undef, size(A,2), size(B,1)),
            Matrix{eltype(A)}(undef, size(A,2), size(B,1)))
end


adi!(F, A, B, C, a, b, c, d; tolerance=1e-15, factorize=factorize) = plan_adi!(A, B, C, a, b, c, d; tolerance=tolerance, factorize=factorize) * F
adi(F, A, B, C, a, b, c, d; tolerance=1e-15, factorize=factorize) = adi!(copy(F), A, B, C, a, b, c, d; tolerance=tolerance, factorize=factorize)

*(P::ADIPlan, F::AbstractMatrix) = P * convert(Matrix, F)
function *(P::ADIPlan, F::Matrix{T}) where T
    p,q,As,Bs,Cfacs,Dfacs,X,Y = P.p,P.q,P.As,P.Bs,P.Cfacs,P.Dfacs,P.tmp1,P.tmp2
    J = length(P.p)
    for j = 1:J
        # (As[j]*X - F/p[j])
        if j ≠ 1
            mul!(Y, As[j], X)
            X .= Y .- F ./ p[j]
        else
            Y .= (-).(F ./ p[j])
        end
        rdiv!(X, Cfacs[j])
        # (X*Bs[j] - F/q[j])
        mul!(Y, X, Bs[j])
        X .= Y .- F ./ q[j]
        ldiv!(Dfacs[j], X)
    end
    X
end

end # module AlternatingDirectionImplicit
