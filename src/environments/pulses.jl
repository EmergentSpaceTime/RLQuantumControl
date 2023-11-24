"""Contains various pulse functions that can modulate the controls (e.g. by
exponential or sinusoidal functions) or inject noise. These pulse functions can
be chained together to create more complex pulses.
"""

abstract type PulseFunction <: Function end

reset!(::PulseFunction, ::AbstractRNG = default_rng()) = nothing

Base.length(::PulseFunction) = nothing


function Base.length(n::Chain{<:Tuple{Vararg{PulseFunction}}})
    ∑ₗ = 0
    ∑ₙ = 0
    for c in n
        l = length(c)
        if !isnothing(l)
            ∑ₙ += 1
            ∑ₗ += l
        end
    end
    μ = ∑ₗ ÷ ∑ₙ
    for c in n
        if !isnothing(length(c))
            if length(c) != μ
                throw(
                    DimensionMismatch(
                        "All noise functions must have the same length "
                        * "paramters (corresponding to a noise on each pulse)."
                    )
                )
            end
        end
    end
    return μ
end

function reset!(
    p::Chain{<:Tuple{Vararg{PulseFunction}}}, rng::AbstractRNG = default_rng()
)
    for c in p
        reset!(c, rng)
    end
    return nothing
end

function (n::Chain{<:Tuple{Vararg{PulseFunction}}})(ϵₜ::Vector{Float64})
    for c in n
        ϵₜ = c(ϵₜ)
    end
    return ϵₜ
end


"""Exponential pulse function."""
struct ExponentialPulse <: PulseFunction end

(::ExponentialPulse)(ϵₜ::Vector{Float64}) = exp.(ϵₜ)


"""Quasi-static Gaussian noise.

Constructs callable that generates an episodic Gaussian noise and adds it to the
pulse.

```math
𝒫(ϵₜ) = ϵₜ + δ; δ ∼ 𝒩(0, σ) ∀ t ∈ [1, T]
```

Args:
  * ϵₙ: Number of controls.
  * σ: Standard deviation of Gaussian noise.

Fields:
  * σ: Noise standard deviation.
"""
struct GaussianNoise <: PulseFunction
    σ::Float64
    _noise_episode::Vector{Float64}
end

GaussianNoise(ϵₙ::Int, σ::Real) = GaussianNoise(σ, zeros(ϵₙ))

Base.length(n::GaussianNoise) = length(n._noise_episode)

function reset!(n::GaussianNoise, rng::AbstractRNG = default_rng())
    n._noise_episode .= randn(rng, length(n._noise_episode)) .* n.σ
    return nothing
end

(n::GaussianNoise)(ϵₜ::Vector{Float64}) = ϵₜ .+ n._noise_episode


"""White Gaussian noise.

Constructs callable that generates a discrete-time Gaussian noise sequence and
adds it to the pulse at each time step.

```math
𝒫(ϵₜ) = ϵₜ + δₜ; δₜ ∼ 𝒩(0, σ)
```

Args:
  * ϵₙ: Number of controls.
  * t̃ₙ: Number of total time steps (including sub-time steps if shaping includes
        oversampling).
  * σ: Standard deviation of additive white Gaussian noise.

Fields:
  * σ: Noise standard deviation.
"""
struct WhiteNoise <: PulseFunction
    σ::Float64
    _time_step::RefValue{Int}
    _noises_episode::Matrix{Float64}
end

function WhiteNoise(ϵₙ::Int, t̃ₙ::Int, σ::Real)
    if t̃ₙ <= 0
        throw(ArgumentError("Number of time steps must be greater than 0!"))
    end
    return WhiteNoise(σ, RefValue(0), zeros(ϵₙ, t̃ₙ))
end

Base.length(n::WhiteNoise) = size(n._noises_episode, 1)

function reset!(n::WhiteNoise, rng::AbstractRNG = default_rng())
    n._time_step[] = 0
    n._noises_episode .= randn(rng, size(n._noises_episode)...) .* n.σ
    return nothing
end

function (n::WhiteNoise)(ϵₜ::Vector{Float64})
    n._time_step[] += 1
    return ϵₜ .+ n._noises_episode[:, n._time_step[]]
end


"""Coloured power noise (∝ 1 / fᵅ).

Constructs callable that generates an episodic Gaussian noise.

```math
𝒫(ϵₜ) = ϵₜ + δₜ; δₜ ∼ 𝒞(S₀, α)[t]
```

Args:
  * ϵₙ: Number of controls.
  * t̃ₙ: Number of total time steps (including sub-time steps if shaping includes
        oversampling).
  * S₀: Noise power constant.
  * α: Noise power exponent.

Fields:
  * S₀: Noise power constant.
  * α: Noise power exponent.
"""
struct ColouredNoise <: PulseFunction
    S₀::Float64
    α::Float64
    _time_step::RefValue{Int}
    _noises_episode::Matrix{Float64}
end

function ColouredNoise(ϵₙ::Int, t̃ₙ::Int, S₀::Real, α::Real)
    if α < 0
        throw(ArgumentError("α must be >= 0"))
    end
    if t̃ₙ <= 0
        throw(ArgumentError("Number of time steps must be greater than 0!"))
    end
    return ColouredNoise(S₀, α, RefValue(0), zeros(ϵₙ, t̃ₙ))
end

Base.length(n::ColouredNoise) = size(n._noises_episode, 1)

function reset!(n::ColouredNoise, rng::AbstractRNG = default_rng())
    n._time_step[] = 0
    n._noises_episode .= power_noise(
        size(n._noises_episode, 2), n.α, n.S₀, size(n._noises_episode, 1), rng
    )
    return nothing
end

function (n::ColouredNoise)(ϵₜ::Vector{Float64})
    n._time_step[] += 1
    return ϵₜ .+ n._noises_episode[:, n._time_step[]]
end
