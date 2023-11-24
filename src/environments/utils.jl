"""Contains various utility functions."""

"""Generates a random unitary.

Generates a random matrix from the group U(n) with probability distributions
given by the respective invariant measures [1].

Args:
  * n: The dimension of the unitary matrix.
  * T: Floating (real) type of the matrix (default: [Float64](@ref)).
  * rng: The random number generator to use (default: [default_rng()](@ref)).

Returns:
  * Matrix{Complex{T}}: A random unitary matrix.

[1] Mezzadri, F. *How to generate random matrices from the classical compact
groups*. [arXiv: 0609050v2](https://arxiv.org/abs/math-ph/0609050) (2006).
"""
function rand_unitary(n::Int, rng::AbstractRNG = default_rng())
    M = (randn(rng, n, n) + randn(rng, n, n) * im) / sqrt(2)
    Q, R = qr(M)
    Λ = diagm(sign.(diag(R)))
    return Q * Λ
end

function rand_unitary(
    n::Int, T::Type{<:AbstractFloat}, rng::AbstractRNG = default_rng()
)
    return convert(Complex{Matrix{T}}, rand_unitary(rng, n))
end


"""Checks if a given matrix is unitary upto floating point precision."""
function is_unitary(u::AbstractMatrix{<:Number})
    return isapprox(u * u', Matrix{eltype(u)}(I, size(u)))
end


"""Fidelity between two unitary gates.

Calculates the fidelity of two unitary gates:

```math
ℱ = |tr(ŪV) / N|
```

Args:
  * u: Unitary matrix.
  * v: Unitary matrix.

Returns:
  * Real: The fidelity (overlap) between two unitary matrices.
"""
function gate_fidelity(u::AbstractMatrix{<:Number}, v::AbstractMatrix{<:Number})
    ∑ = zero(promote_type(eltype(u), eltype(v)))
    @simd for e in eachindex(v)
        @inbounds ∑ += conj(u[e]) * v[e]
    end
    return abs(∑ / size(u, 1))
end


"""Generate coloured-noise samples with arbitrary power.

Generates a power-law noise in the time domain of spectral density [1]:

```math
S ∝ 1 / fᵅ
```

Args:
  * n: Length of time series to generate noise.
  * α: Power of frequency (e.g. 0 for Gaussian noise and 1 for pink noise).
  * scale: Scaling of density factor of the noise (default: 1.0).
  * k: Number of independent noise sequences (default: 1).
  * T: Floating type of the noise (default: [Float64](@ref)).
  * rng: The random number generator to use (default: [default_rng()](@ref)).

Kwargs:
  * normalise: Whether to normalise the noise to unity (default: false).

Returns:
  * Matrix{Float64}: A matrix of size (k, n) of coloured noise samples.

[1] Timmer, J. and Koenig, M. *On generating power law noise*. A & A, 300, 707
(1995). https://ui.adsabs.harvard.edu/abs/1995A&A...300..707T
"""
function power_noise(
    n::Int,
    α::Real,
    scale::Real = 1.0,
    k::Int = 1,
    rng::AbstractRNG = default_rng();
    normalise::Bool = false,
)
    n_half = floor(Int, n / 2) - 1

    f = unsqueeze(collect(UnitRange(1, n_half)); dims=1) ./ n

    power_half = @. 1 / f ^ (α / 2)
    noise_half = @. $randn(rng, k, n_half) + $randn(rng, k, n_half) * im
    shaped_half = power_half .* noise_half

    shaped = hcat(
        ones(k),
        shaped_half,
        fill(1 / ((n_half + 1) / n) ^ (α / 2), k),
        reverse(conj(shaped_half); dims=2),
    )
    noise_t = real(ifft(sqrt(scale / 2) .* shaped, 2))
    if normalise
        return @. noise_t / sqrt($mean(noise_t ^ 2, dims=2))
    end
    return noise_t
end

function power_noise(
    n::Int,
    α::Real,
    scale::Real,
    T::Type{<:AbstractFloat},
    k::Int = 1,
    rng::AbstractRNG = default_rng();
    normalise::Bool = true,
)
    return convert(
        Matrix{T}, power_noise(n, α, scale, k, rng; normalise = normalise)
    )
end
