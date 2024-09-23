"""Abstract callable struct that can modulate the controls (e.g. by exponential
or sinusoidal functions) or inject noise. Custom pulse functions may implement
optional [`reset!`]() and [`has_noise`]() methods if required.

These callables have the argument signature:
```math
    \\mathscr{P}(t, \\epsilon_{t})\\rightarrow\\epsilon_{t}'
```
"""
abstract type PulseFunction <: Function end

reset!(::PulseFunction, ::AbstractRNG = default_rng()) = nothing
has_noise(::PulseFunction) = false
_n_ctrls(::PulseFunction) = nothing
_n_ts(::PulseFunction) = nothing


"""
    IdentityPulse()

Identity pulse function.
```math
    \\mathscr{P}(t, \\epsilon_{t}) = \\epsilon_{t}
```
"""
struct IdentityPulse <: PulseFunction end

(::IdentityPulse)(::Int, epsilon_t::Vector{Float64}) = epsilon_t


"""
    ExponentialPulse()

Exponential pulse function.
```math
    \\mathscr{P}(t, \\epsilon_{t}) = \\exp{(\\epsilon_{t})}
```
"""
struct ExponentialPulse <: PulseFunction end

(::ExponentialPulse)(::Int, epsilon_t::Vector{Float64}) = exp.(epsilon_t)


struct StaticNoiseInjection <: PulseFunction
    sigma::Float64
    _noise_episode::Vector{Float64}
end

"""
    StaticNoiseInjection(n_controls::Int, sigma::Real)

Callable that injects an episodic Gaussian noise to the pulse:

```math
    \\mathscr{P}(t, \\epsilon_{t}) = \\epsilon_{t} + \\delta
```
Where:
```math
    \\delta\\sim\\mathscr{N}(0, \\sigma)\\ \\ \\forall t \\in [1, T]
```

Args:
  * `n_controls`: Number of controls.
  * `sigma`: Standard deviation of Gaussian noise.

Fields:
  * `sigma`: Noise standard deviation.
"""
function StaticNoiseInjection(n_controls::Int, sigma::Real)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    sigma < 0 && throw(ArgumentError("`sigma` must be >= 0."))
    return StaticNoiseInjection(sigma, zeros(n_controls))
end

function (n::StaticNoiseInjection)(::Int, epsilon_t::Vector{Float64})
    return epsilon_t .+ n._noise_episode
end

function reset!(n::StaticNoiseInjection, rng::AbstractRNG = default_rng())
    n._noise_episode .= randn(rng, length(n._noise_episode)) .* n.sigma
    return nothing
end

has_noise(::StaticNoiseInjection) = true
_n_ctrls(n::StaticNoiseInjection) = length(n._noise_episode)


struct WhiteNoiseInjection <: PulseFunction
    sigma::Float64
    _noises_episode::Matrix{Float64}
end

"""
    WhiteNoiseInjection(n_controls::Int, n_ts::Int, sigma::Real)

Callable that generates uncorrelated noise on each time step of the incoming
pulse:
```math
    \\mathscr{P}(t, \\epsilon_{t}) = \\epsilon_{t} + \\delta_{t}
```
Where:
```math
    \\delta_{t}\\sim\\mathscr{N}(0, \\sigma)[t]
```

Args:
  * `n_controls`: Number of controls.
  * `n_ts`: Number of total time steps (including extra time steps if shaping
        includes oversampling and boundary conditions).
  * `sigma`: Standard deviation of additive white Gaussian noise.

Fields:
  * `sigma`: Noise standard deviation.
"""
function WhiteNoiseInjection(n_ts::Int, n_controls::Int, sigma::Real)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    n_ts < 1 && throw(ArgumentError("`n_ts` must be >= 1."))
    sigma < 0 && throw(ArgumentError("`sigma` must be >= 0."))
    return WhiteNoiseInjection(sigma, Base.RefValue(0), zeros(n_controls, n_ts))
end

function (n::WhiteNoiseInjection)(t_step::Int, epsilon_t::Vector{Float64})
    return epsilon_t .+ n._noises_episode[:, t_step]
end

function reset!(n::WhiteNoiseInjection, rng::AbstractRNG = default_rng())
    n._noises_episode .= randn(rng, size(n._noises_episode)...) .* n.sigma
    return nothing
end

has_noise(::WhiteNoiseInjection) = true
_n_ctrls(n::WhiteNoiseInjection) = size(n._noises_episode, 1)
_n_ts(n::WhiteNoiseInjection) = size(n._noises_episode, 2)


struct ColouredNoiseInjection <: PulseFunction
    s_0::Float64
    alpha::Float64
    _noises_episode::Matrix{Float64}
end

"""
    ColouredNoiseInjection(n_controls::Int, n_ts::Int, s_0::Real, alpha::Real)

Callable that generates time-correlated noise on each time step of the incoming
pulse:
```math
    \\mathscr{P}(t, \\epsilon_{t}) = \\epsilon_{t} + \\delta_{t}
```
Where:
```math
    \\delta_{t}\\sim\\mathscr{C}(S_{0}, \\alpha)[t]
```
```math
    \\mathscr{C}(S_{0}, \\alpha) = \\frac{S_{0}}{f^{\\alpha}}
```

Args:
  * `n_controls`: Number of controls.
  * `n_ts`: Number of total time steps (including extra time steps if shaping
        includes oversampling).
  * `s_0`: Noise power constant.
  * `alpha`: Noise power exponent.

Fields:
  * `s_0`: Noise power constant.
  * `alpha`: Noise power exponent.
"""
function ColouredNoiseInjection(
    n_controls::Int, n_ts::Int, s_0::Real, alpha::Real
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    n_ts < 1 && throw(ArgumentError("`n_ts` must be >= 1."))
    if alpha <= 0
        return throw(
            ArgumentError(
                "`alpha` must be > 0. For white noise (`alpha` = 0) use the"
                * " `WhiteNoiseInjection` struct."
            )
        )
    end
    return ColouredNoiseInjection(s_0, alpha, zeros(n_controls, n_ts))
end

function (n::ColouredNoiseInjection)(t_step::Int, epsilon_t::Vector{Float64})
    return epsilon_t .+ n._noises_episode[:, t_step]
end

function reset!(n::ColouredNoiseInjection, rng::AbstractRNG = default_rng())
    n._noises_episode .= power_noise(
        size(n._noises_episode, 2),
        size(n._noises_episode, 1),
        n.alpha,
        n.s_0,
        rng,
    )
    return nothing
end

has_noise(::ColouredNoiseInjection) = true
_n_ctrls(n::ColouredNoiseInjection) = size(n._noises_episode, 1)
_n_ts(n::ColouredNoiseInjection) = size(n._noises_episode, 2)


function (c_p::Chain{<:Tuple{Vararg{PulseFunction}}})(
    t_step::Int, epsilon_t::Vector{Float64}
)
    for p in c_p
        epsilon_t = p(t_step, epsilon_t)
    end
    return epsilon_t
end

function reset!(
    c_p::Chain{<:Tuple{Vararg{PulseFunction}}}, rng::AbstractRNG = default_rng()
)
    for p in c_p
        reset!(p, rng)
    end
    return nothing
end

function has_noise(c_p::Chain{<:Tuple{Vararg{PulseFunction}}})
    for p in c_p
        has_noise(p) && return true
    end
    return false
end

function _n_ctrls(c_p::Chain{<:Tuple{Vararg{PulseFunction}}})
    sum_length = 0
    sum_chain = 0
    for p in c_p
        l = _n_ctrls(p)
        if !isnothing(l)
            sum_chain += 1
            sum_length += l
        end
    end
    iszero(sum_chain) && return nothing
    equal_length = sum_length ÷ sum_chain
    for p in c_p
        if !isnothing(_n_ctrls(p))
            if _n_ctrls(p) != equal_length
                throw(
                    DimensionMismatch(
                        "All pulses functions must have the same control length"
                        * " parameters (corresponding to a noise on each pulse)"
                        * "."
                    )
                )
            end
        end
    end
    return equal_length
end

function _n_ts(c_p::Chain{<:Tuple{Vararg{PulseFunction}}})
    sum_length = 0
    sum_chain = 0
    for p in c_p
        l = _n_ts(p)
        if !isnothing(l)
            sum_chain += 1
            sum_length += l
        end
    end
    iszero(sum_chain) && return nothing
    equal_length = sum_length ÷ sum_chain
    for p in c_p
        if !isnothing(_n_ts(p))
            if _n_ts(p) != equal_length
                throw(
                    DimensionMismatch(
                        "All pulses functions must have the same time-step"
                        * " length parameters (corresponding to a noise on each"
                        * " pulse)."
                    )
                )
            end
        end
    end
    return equal_length
end
