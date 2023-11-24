"""Contains various pulse shaping types, to smooth out discrete piece-wise
constant pulses.
"""

abstract type ShapingFunction <: Function end


"""Identity shaping function that does not change the piece-wise pulses.

Args:
  * ϵₙ: Number of controls.
  * tₙ: Number of time steps per episode (equal to number of inputted actions).

Fields:
  * pulse_history: History of input pulses.
"""
struct IdentityShaping <: ShapingFunction
    pulse_history::Matrix{Float64}
    _time_step::RefValue{Int}
end

IdentityShaping(ϵₙ::Int, tₙ::Int) = IdentityShaping(zeros(ϵₙ, tₙ), RefValue(0))

function reset!(s::IdentityShaping)
    s._time_step[] = 0
    s.pulse_history .= zero(s.pulse_history)
    return nothing
end

function (s::IdentityShaping)(ϵₜ::AbstractVector{Float64})
    s._time_step[] += 1
    s.pulse_history[:, s._time_step[]] = ϵₜ
    return s.pulse_history[:, s._time_step[]]
end


"""Gaussian shaping filter.

Creates a callable that can be used to convolute piece-wise constant pulses with
a Gaussian of given standard deviation.

```math
𝒮(ϵₜ) = ϵ̄ₜ = ∑ₙϵₕ[n]𝒫[t - n]
```

Args:
  * ϵₙ: Number of controls.
  * tₙ: Number of time steps per episode (equal to number of inputted actions).
  * σ: The standard deviation.

Kwargs:
  * sampling_rate: The (over-)sampling rate, i.e. the number of sub-steps per
        discrete piece-wise pulse step (default: 10).

Fields:
  * pulse_history: History of input pulses.
  * shaped_pulse_history: History of shaped pulses.
  * σ: The standard deviation.
  * sampling_rate: Number of sub-steps per discrete piece-wise pulse step.
"""
struct GaussianShaping <: ShapingFunction
    pulse_history::Matrix{Float64}
    shaped_pulse_history::Matrix{Float64}
    sampling_rate::Int
    σ::Float64
    _kernel::Spline1D
    _time_step::RefValue{Int}
end

function GaussianShaping(
    ϵₙ::Int, tₙ::Int, σ::Real, μ::Real; sampling_rate::Int = 10
)
    if sampling_rate < 1
        throw(ArgumentError("sampling_rate must be >= 1"))
    end
    return GaussianShaping(
        zeros(ϵₙ, tₙ),
        zeros(ϵₙ, tₙ * sampling_rate),
        sampling_rate,
        σ,
        Spline1D(
            -2.9:0.05:7.1,
            @. (
                1
                / (σ * sqrt(2π))
                * exp(-0.5 * (($collect(-2.9:0.05:7.1) - μ) / σ) ^ 2)
            );
            bc="zero",
        ),
        RefValue(0),
    )
end

function reset!(s::GaussianShaping)
    s._time_step[] = 0
    s.pulse_history .= zero(s.pulse_history)
    s.shaped_pulse_history .= zero(s.shaped_pulse_history)
    return nothing
end

function (s::GaussianShaping)(ϵₜ::AbstractVector{Float64})
    s._time_step[] += 1
    s.pulse_history[:, s._time_step[]] = ϵₜ

    t_sub = range(
        s.sampling_rate * (s._time_step[] - 1) + 1,
        s.sampling_rate * s._time_step[]
    )
    ϵ̄ₜ = zeros(length(ϵₜ), s.sampling_rate)
    for n in 1:s._time_step[]
        for (i, t) in enumerate(t_sub)
            ϵ̄ₜ[:, i] += (
                s.pulse_history[:, n]
                * s._kernel((t - n * s.sampling_rate) / s.sampling_rate)
            )
        end
    end

    s.shaped_pulse_history[:, t_sub] = ϵ̄ₜ
    return ϵ̄ₜ
end


"""Impulse response filter.

Creates a callable that can be used to convolute piece-wise constant pulses with
a given impulse response.

```math
𝒮(ϵₜ) = ϵ̄ₜ = ∑ₙϵₕ[n]k[t - n]
```

Args:
  * ϵₙ: Number of controls.
  * tₙ: Number of time steps per episode (equal to number of inputted actions).
  * response_data: A text file with x(time)-y coordinates to generate a spline.

Kwargs:
  * sampling_rate: The (over-)sampling rate, i.e. the number of sub-steps per
        discrete piece-wise pulse step (default: 10).
  * smoothing: Smoothing parameter for spline (default: 0.0).

Fields:
  * pulse_history: History of input pulses.
  * shaped_pulse_history: History of shaped pulses.
  * kernel: Spline to smooth out pulse.
  * sampling_rate: Number of sub-steps per discrete piece-wise pulse step.
"""
struct ImpulseResponseFilter <: ShapingFunction
    pulse_history::Matrix{Float64}
    shaped_pulse_history::Matrix{Float64}
    kernel::Spline1D
    sampling_rate::Int
    _time_step::RefValue{Int}
end

function ImpulseResponseFilter(
    ϵₙ::Int,
    tₙ::Int,
    response_data::String;
    sampling_rate::Int = 10,
    smoothing::Real = 0.0,
)
    if sampling_rate < 1
        throw(ArgumentError("sampling_rate must be >= 1"))
    end

    d = readdlm(response_data)
    @. d[:, 2] = (
        2 * ($maximum(d[:, 1]) - $minimum(d[:, 1])) * d[:, 2] / $sum(d[:, 2])
    )
    return ImpulseResponseFilter(
        zeros(ϵₙ, tₙ),
        zeros(ϵₙ, tₙ * sampling_rate),
        Spline1D(d[:, 1], d[:, 2], s=smoothing, bc="zero"),
        sampling_rate,
        RefValue(0),
    )
end

function reset!(s::ImpulseResponseFilter)
    s._time_step[] = 0
    s.pulse_history .= zero(s.pulse_history)
    s.shaped_pulse_history .= zero(s.shaped_pulse_history)
    return nothing
end

function (s::ImpulseResponseFilter)(ϵₜ::AbstractVector{Float64})
    s._time_step[] += 1
    s.pulse_history[:, s._time_step[]] = ϵₜ

    t_sub = range(
        s.sampling_rate * (s._time_step[] - 1) + 1,
        s.sampling_rate * s._time_step[]
    )
    ϵ̄ₜ = zeros(length(ϵₜ), s.sampling_rate)
    for n in 1:s._time_step[]
        for (i, t) in enumerate(t_sub)
            ϵ̄ₜ[:, i] += (
                s.pulse_history[:, n]
                * s.kernel((t - n * s.sampling_rate) / s.sampling_rate)
            )
        end
    end

    s.shaped_pulse_history[:, t_sub] = ϵ̄ₜ
    return ϵ̄ₜ
end
