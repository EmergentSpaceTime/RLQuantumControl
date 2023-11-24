"""Contains various observation types. Can add more complicated observations
such as partial observability with expectation values, etc.
"""

abstract type ObservationFunction <: Function end


"""Full state observation including time, pulse, and unitary."""
struct FullObservation <: ObservationFunction end

(::FullObservation)(state::Vector) = state


"""Partial state observations.

Args:
* ϵₙ: Number of control fields.
* dᵤ: Unitary dimension if 'pulse' not selected (default: nothing).

Kwargs:
  * observation_type: The elements of the state that can be observed. Options
        are 'pulse' for only time and pulse observations, 'unitary' for time and
        unitary matrix, and 'triu' for time and minimal unitary parameters
        (default: 'pulse').
"""
struct PartialObservation <: ObservationFunction
    _observation_indices::Vector{Int}
end

function PartialObservation(
    ϵₙ::Int,
    dᵤ::Union{Nothing, Int} = nothing;
    observation_type::String = "pulse",
)
    if observation_type == "pulse"
        _observation_indices = collect(1 : 1 + ϵₙ)
    elseif observation_type == "unitary"
        _observation_indices = vcat(1, collect(2 + ϵₙ : 1 + ϵₙ + 2 * dᵤ ^ 2))
    elseif observation_type == "triu"
        _observation_indices = sort(
            vcat(
                1,
                [2 * (i + (j - 1) * dᵤ) for j in 1:dᵤ for i in 1:j] .+ 1 .+ ϵₙ,
                [2 * (i + (j - 1) * dᵤ) for j in 1:dᵤ for i in 1:j] .+ ϵₙ,
            )
        )
    else
        throw(
            ArgumentError(
                "Chosen partial observation style not supported. Options are"
                * "'pulse' for only time and pulse observations, 'unitary' for"
                * "time and unitary matrix, and 'triu' for time and minimal"
                * "unitary parameters."
            )
        )
    end
    return PartialObservation(_observation_indices)
end

(o::PartialObservation)(state::Vector) = state[o._observation_indices]


"""Observation function with values normalised to 𝒩(0, 1).

Args:
  * base_function: Base observation function.
  * dₒ: Observation function output dimension.

Fields:
  * base_function: Base observation function.
  * observations_μ: Observations mean.
  * observations_σ²: Observations variance.
  * count: The number of seen observations.
"""
struct NormalisedObservation{𝒪 <: ObservationFunction} <: ObservationFunction
    base_function::𝒪
    observations_μ::Vector{Float64}
    observations_σ²::Vector{Float64}
    count::RefValue{Int}
end

function NormalisedObservation(base_function::ObservationFunction, dₒ::Int)
    if base_function isa NormalisedObservation
        throw(
            ArgumentError(
                "Base observation function can't already be the normalised"
                * "observation function!"
            )
        )
    end
    return NormalisedObservation(
        base_function, zeros(Float64, dₒ), ones(Float64, dₒ), RefValue(1)
    )
end

function (n::NormalisedObservation)(state::Vector{Float64})
    observation = n.base_function(state)
    Δₒ = observation .- n.observations_μ
    @. n.observations_μ += Δₒ / (n.count[] + 1)
    @. n.observations_σ² = (
        n.observations_σ² * n.count[] / (n.count[] + 1)
        + (Δₒ ^ 2) * n.count[] / (n.count[] + 1) ^ 2
    )
    n.count[] += 1
    return @. (
        (observation - n.observations_μ)
        / sqrt(n.observations_σ² + eps(Float64))
    )
end

function (n::NormalisedObservation)(
    state_space::Vector{ClosedInterval{Float64}}
)
    return n.base_function(state_space)
end
