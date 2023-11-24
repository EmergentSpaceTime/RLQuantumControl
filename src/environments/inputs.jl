"""Contains various input types to the controller (e.g. step-change or direct)
that creates a output pulse from a given action.
"""

abstract type InputFunction <: Function end


"""Step input function.

Creates a callable that takes an action and changes the current pulse by a
number in (-Δϵ, Δϵ) for continuous actions or (±1 or 0) * Δϵ for discrete
actions, but within the bounds of (ϵₘᵢₙ, ϵₘₐₓ).

```math
ℐ(ϵₜ₋₁, aₜ) = ϵₜ₋₁ + aₜ
```

Args:
  * ϵₙ: Number of controls.

Kwargs:
  * ϵₘᵢₙ: Minimum pulse amplitude (default: fill(-Inf, ϵₙ)).
  * ϵₘₐₓ: Maximum pulse amplitude (default: fill(Inf, ϵₙ)).
  * Δϵ: Maximal change in pulse amplitude (default: fill(Inf, ϵₙ)).

Fields:
  * ϵₙ: Number of controls.
  * ϵₘᵢₙ: Minimum pulse amplitudes.
  * ϵₘₐₓ: Maximum pulse amplitudes.
  * Δϵ: Maximal change in pulse amplitudes.
"""
struct StepInput <: InputFunction
    ϵₙ::Int
    ϵₘᵢₙ::Vector{Float64}
    ϵₘₐₓ::Vector{Float64}
    Δϵ::Vector{Float64}
end

function StepInput(
    ϵₙ::Int;
    ϵₘᵢₙ::Vector{<:Real} = fill(-Inf, ϵₙ),
    ϵₘₐₓ::Vector{<:Real} = fill(Inf, ϵₙ),
    Δϵ::Vector{<:Real} = fill(Inf, ϵₙ),
)
    if (ϵₙ != length(ϵₘᵢₙ)) & (ϵₙ != length(ϵₘₐₓ)) & (ϵₙ != length(Δϵ))
        throw(
            DimensionMismatch(
                "Length of ϵₘᵢₙ, ϵₘₐₓ, and Δϵ must be equal to ϵₙ."
            )
        )
    end
    return StepInput(ϵₙ, ϵₘᵢₙ, ϵₘₐₓ, Δϵ)
end

function (i::StepInput)(ϵₜ₋₁::AbstractVector{Float64}, a::Vector{Float64})
    return clamp(ϵₜ₋₁ + a, i.ϵₘᵢₙ, i.ϵₘₐₓ)
end

function (i::StepInput)(ϵₜ₋₁::AbstractVector{Float64}, a::Int)
    aᵥ = @. ($digits(a - 1; base=3, pad=i.ϵₙ) - 1) * i.Δϵ
    return clamp(ϵₜ₋₁ + aᵥ, i.ϵₘᵢₙ, i.ϵₘₐₓ)
end


"""Direct input function.

Creates a callable that takes an action and outputs a pulse in (ϵₘᵢₙ, ϵₘₐₓ) for
continuous actions or ϵₘᵢₙ / ϵₘₐₓ for discrete actions.

```math
ℐ(ϵₜ₋₁, aₜ) = aₜ
```

Args:
  * ϵₙ: Number of controls.

Kwargs:
  * ϵₘᵢₙ: Minimum pulse amplitude (default: fill(-Inf, ϵₙ)).
  * ϵₘₐₓ: Maximum pulse amplitude (default: fill(Inf, ϵₙ)).

Fields:
  * ϵₙ: Number of control pulses.
  * ϵₘᵢₙ: Minimum pulse amplitudes.
  * ϵₘₐₓ: Maximum pulse amplitudes.
"""
struct DirectInput <: InputFunction
    ϵₙ::Int
    ϵₘᵢₙ::Vector{Float64}
    ϵₘₐₓ::Vector{Float64}
end

function DirectInput(
    ϵₙ::Int,
    ϵₘᵢₙ::Vector{<:Real} = fill(-Inf, ϵₙ),
    ϵₘₐₓ::Vector{<:Real} = fill(Inf, ϵₙ),
)
    if (ϵₙ != length(ϵₘᵢₙ)) & (ϵₙ != length(ϵₘₐₓ))
        throw(DimensionMismatch("Length of ϵₘᵢₙ and ϵₘₐₓ must be equal to ϵₙ."))
    end
    return DirectInput(ϵₙ, ϵₘᵢₙ, ϵₘₐₓ)
end

function (::DirectInput)(::AbstractVector{Float64}, a::Vector{Float64})
    return a
end

function (i::DirectInput)(::AbstractVector{Float64}, a::Int)
    pulse = digits(a - 1; base=2, pad=i.ϵₙ)
    return i.ϵₘᵢₙ + (i.ϵₘₐₓ - i.ϵₘᵢₙ) * pulse
end
