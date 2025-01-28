"""
    rand_unitary(n::Int, rng::AbstractRNG = default_rng())

Generates a random matrix from the group ``U(n)`` with probability distributions
given by the respective invariant measures [mezzadri2007generate](@cite).

Args:
  * `n`: The dimension of the unitary matrix.
  * `rng`: The random number generator to use (default:
        [`Random.default_rng()`]()).

Returns:
  * `Matrix{ComplexF64}`: A random unitary matrix.
"""
function rand_unitary(n::Int, rng::AbstractRNG = default_rng())
    m = (randn(rng, n, n) + randn(rng, n, n) * im) / sqrt(2)
    q, r = qr(m)
    l = diagm(sign.(diag(r)))
    return q * l
end


"""
    is_unitary(u::AbstractMatrix)

Checks if a given matrix is unitary up to floating point precision.
"""
is_unitary(u::AbstractMatrix) = isapprox(u * u', Matrix{eltype(u)}(I, size(u)))


"""
    gate_fidelity(u::AbstractMatrix, v::AbstractMatrix)

Calculates the (unsquared) fidelity of two unitary gates:
```math
    \\mathscr{F}(U, V) = \\left|\\frac{\\text{tr}(U^{\\dagger}V)}{N}\\right|
```

Args:
  * `u`: Unitary matrix.
  * `v`: Unitary matrix.

Returns:
  * `Real`: The fidelity (overlap) between two unitary matrices.
"""
function gate_fidelity(u::AbstractMatrix, v::AbstractMatrix)
    sum_of_elements = zero(promote_type(eltype(u), eltype(v)))
    @simd for e in eachindex(v)
        @inbounds sum_of_elements += conj(u[e]) * v[e]
    end
    return abs(sum_of_elements / size(u, 1))
end


"""
    power_noise(
        n::Int,
        k::Int,
        alpha::Real,
        f_s::Real = 1.0,
        scale::Real = 1.0,
        rng::AbstractRNG = default_rng();
        normalise::Bool = false,
    )

Generates a power-law noise in the time domain with spectral density
[1995A&A...300..707T](@cite):
```math
    S\\propto\\frac{1}{f^{\\alpha}}
```

Args:
  * `n`: Length of time series to generate noise.
  * `k`: Number of independent noise sequences.
  * `alpha`: Power of frequency (e.g. `0` for Gaussian noise).
  * `f_s`: Sampling frequency (default: `1.0`).
  * `scale`: Scaling of density factor of the noise (default: `1.0`).
  * `rng`: The random number generator to use (default:
        [`Random.default_rng()`]()).

Kwargs:
  * `normalise`: Whether to normalise the noise to unity (default: `false`).

Returns:
  * `Matrix{Float64}`: A matrix of size (k, n) of coloured noise samples.
"""
function power_noise(
    n::Int,
    k::Int,
    alpha::Real,
    f_s::Real = 1.0,
    scale::Real = 1.0,
    rng::AbstractRNG = default_rng();
    normalise::Bool = false,
)
    g_noise = rfft(randn(rng, k, n), 2)
    f_powers = unsqueeze(
        _psd.(rfftfreq(n, f_s), alpha, scale * (f_s / 2)); dims=1
    )
    noise_t = irfft(g_noise .* f_powers, n, 2)
    normalise && return @. noise_t / sqrt($mean(noise_t ^ 2, dims=2))
    return noise_t
end


function _psd(f::Real, alpha::Real, scale::Real)
    if iszero(f)
        return zero(Float64)
    end
    return sqrt(scale) * (1 / f) ^ (alpha / 2)
end
