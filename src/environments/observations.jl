"""Abstract callable struct that encodes the "measurement" information the agent
can recieve to decide actions. Custom observations should define an
[`observation_space`]() method.

These callables have the argument signature:
```math
    \\mathscr{O}(s_{t})
```
"""
abstract type ObservationFunction <: Function end


"""
    FullObservation()

Full state observation including time-to-go, pulse, and evolution operator.
"""
struct FullObservation <: ObservationFunction end

(::FullObservation)(state::Vector{Float64}) = state

function observation_space(
    ::FullObservation, state_space::Vector{ClosedInterval{Float64}}
)
    return state_space
end


struct MinimalObservation <: ObservationFunction
    _observation_indices::Vector{Int}
end

"""
    MinimalObservation(n_controls::Union{Nothing, Int} = nothing)

Minimal observation including only the time-to-go and optionally the pulse.

Args:
  * `n_controls`: Number of control pulses. Leave as `nothing` if using just the
        time-to-go (default: `nothing`).
"""
function MinimalObservation(n_controls::Union{Nothing, Int} = nothing)
    isnothing(n_controls) && return MinimalObservation([1])
    if n_controls <= 0
        throw(ArgumentError("`n_controls` must be a positive integer."))
    end
    return MinimalObservation(collect(1 : 1 + n_controls))
end

(m::MinimalObservation)(state::Vector{Float64}) = state[m._observation_indices]

function observation_space(
    o::MinimalObservation, state_space::Vector{ClosedInterval{Float64}}
)
    return state_space[o._observation_indices]
end


struct ExactTomography <: ObservationFunction
    _observation_indices::Vector{Int}
end

"""
    ExactTomography(
        n_controls::Int,
        observation_type::String,
        process_dim::Int,
        include_pulse::Bool,
        cols::Tuple{Vararg{Int}} = (),
    )

Partial RL state observation with the assumption of perfect statistics about the
quantum process. Interpolation can be done between information about one or all
coloumns of the evolution operator, corresponding to tomography with a specified
input.

Args:
  * `n_controls`: Number of control pulses if including pulse information.
  * `observation_type`: The elements of the process that can be observed.
        Options are `"process"` for time and process matrix, `"triu"` for
        time-to-go and minimal (unitary) parameters, `"partial"` for coloumns of
        matrix (corresponding to tomography with a specified input).
  * `process_dim`: Process matrix dimension.
  * `include_pulse`: Whether to include pulse information (required for delayed
        finite pulses).
  * `cols`: Elements of process matrix to observe for `partial` observation
            types (default: `nothing`).
"""
function ExactTomography(
    n_controls::Int,
    observation_type::String,
    process_dim::Int,
    include_pulse::Bool,
    cols::Tuple{Vararg{Int}} = (),
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))

    first_indices = include_pulse ? collect(1 : 1 + n_controls) : [1]
    if observation_type == "process"
        if include_pulse
            @warn (
                "Consider using `FullObservation` for more efficient"
                * " observation if including pulse information to"
                * " `observation_type` = `process`."
            )
        end
        _observation_indices = vcat(
            first_indices,
            collect(2 + n_controls : 1 + n_controls + 2 * process_dim ^ 2),
        )
    elseif observation_type == "triu"
        _observation_indices = sort(
            vcat(
                first_indices,
                [
                    2 * (j + (i - 1) * process_dim) + n_controls + k
                    for i in 1:process_dim
                    for j in 1:i
                    for k in 0:1
                ],
            )
        )
    elseif observation_type == "partial"
        if isnothing(cols) || iszero(length(cols))
            @warn (
                "Consider using `MinimalObservation` for more efficient"
                * " observation if only including pulse information to"
                * " `observation_type` = `partial` (i.e. `cols` is empty)."
            )
        end
        _observation_indices = vcat(
            first_indices,
            [
                (k - 1) * 2 * process_dim + i + 1 + n_controls
                for k in cols
                for i in 1 : 2 * process_dim
            ],
        )
    else
        throw(
            ArgumentError(
                "Chosen partial observation style not supported. Options are"
                * "`process`, `triu`, and `partial`."
            )
        )
    end
    return ExactTomography(_observation_indices)
end

(o::ExactTomography)(state::Vector{Float64}) = state[o._observation_indices]

function observation_space(
    o::ExactTomography, state_space::Vector{ClosedInterval{Float64}}
)
    return state_space[o._observation_indices]
end


struct UnitaryTomography <: ObservationFunction
    N::Int
    a::Float64
    b::Float64
    _n_controls::Int
    _unitary_dim::Int
    _first_indices::Vector{Int}
    _unitary_indices::Vector{Int}
end

"""
    UnitaryTomography(
        n_controls::Int,
        observation_type::String,
        unitary_dim::Int,
        include_pulse::Bool,
        cols::Tuple{Vararg{Int}} = ();
        N::Int = 10000,
        a::Real = 0.4,
        b::Real = 0.1,
    )

Partial RL state observation sampled from a multinomial distribution of POVM
outcomes with the following POVM and input elements [Baldwin_2014](@cite):
```math
    \\begin{aligned}
        & \\vert \\psi_{0}\\rangle = \\vert 0\\rangle\\\\
        & \\vert \\psi_{n}\\rangle = {%
            \\frac{1}{\\sqrt{2}}
            \\left(\\vert 0\\rangle + \\vert n\\rangle\\right)
            \\ \\ n = 1, \\ldots, d - 1
        }
    \\end{aligned}
```
And:
```math
    \\begin{aligned}
        & E_{0} = a\\vert 0\\rangle\\langle 0\\vert\\\\
        & E_{j} = b\\left(
            1
            + \\vert j\\rangle\\langle 0\\vert
            + \\vert 0\\rangle\\langle j\\vert
        \\right)\\ \\ j = 1, \\ldots, d - 1\\\\
        & \\tilde{E}_{j} = b\\left(
            1
            + i\\vert j\\rangle\\langle 0\\vert
            - i\\vert 0\\rangle\\langle j\\vert
        \\right)\\ \\ j = 1, \\ldots, d - 1\\\\
        & E_{2d} = 1 - E_{0} - \\sum_{j = 1}^{d - 1} E_{j} + \\tilde{E}_{j}
    \\end{aligned}
```
Where ``a, b > 0`` are chosen such that ``E_{2d} > 0``.

Interpolation can be done between information about one or all coloumns of the
evolution operator, corresponding to tomography with a specified input.

Args:
  * `n_controls`: Number of control pulses if including pulse information.
  * `observation_type`: The elements of the unitary that can be observed.
        Options are `"unitary"` for time and unitary matrix, `"triu"` for
        time-to-go and minimal (unitary) parameters, `"partial"` for coloumns of
        matrix (corresponding to tomography with a specified input).
  * `unitary_dim`: Unitary matrix dimension.
  * `include_pulse`: Whether to include pulse information (required for delayed
        finite pulses).
  * `cols`: Elements of unitary matrix to observe for `partial` observation
            types (default: `nothing`).

Kwargs:
  * `N`: Number of samples to draw from the multinomial distribution (default:
            10000).
  * `a`: Real positive number to ensure POVM positivity (default: 0.4).
  * `b`: Real positive number to ensure POVM positivity (default: 0.1).

Fields:
  * `N`: Number of samples to draw from the multinomial distribution.
  * `a`: Real positive number to ensure POVM positivity.
  * `b`: Real positive number to ensure POVM positivity.
"""
function UnitaryTomography(
    n_controls::Int,
    observation_type::String,
    unitary_dim::Int,
    include_pulse::Bool,
    cols::Tuple{Vararg{Int}} = ();
    N::Int = 10000,
    a::Real = 0.4,
    b::Real = 0.1,
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    if (a <= 0) | (b <= 0)
        throw(ArgumentError("`a` and `b` must be positive real numbers."))
    end
    N <= 0 && throw(ArgumentError("`N` must be a positive integer."))
    if N < 1000
        @warn (
            "Consider using a larger `N` for more accurate observation. For"
            * " small `N` there is a chance of numerical instability."
        )
    end

    _first_indices = include_pulse ? collect(1 : 1 + n_controls) : [1]
    if observation_type == "unitary"
        _unitary_indices = collect(1 : 2 * unitary_dim ^ 2)
    elseif observation_type == "triu"
        _unitary_indices = sort(
            [
                2 * (j + (i - 1) * unitary_dim) + k
                for i in 1:unitary_dim
                for j in 1:i
                for k in -1:0
            ],
        )
    elseif observation_type == "partial"
        if isnothing(cols) || iszero(length(cols))
            @warn (
                "Consider using `MinimalObservation` for more efficient"
                * " observation if only including pulse information to"
                * " `observation_type` = `partial` (i.e. `cols` is empty)."
            )
        end
        _unitary_indices = sort(
            [
                (k - 1) * 2 * unitary_dim + i
                for k in cols
                for i in 1 : 2 * unitary_dim
            ],
        )
    else
        throw(
            ArgumentError(
                "Chosen partial observation style not supported. Options are"
                * "`unitary`, `triu`, and `partial`."
            )
        )
    end
    return UnitaryTomography(
        N, a, b, n_controls, unitary_dim, _first_indices, _unitary_indices
    )
end

function (o::UnitaryTomography)(state::Vector{Float64})
    p = _get_probabilities_from_state(
        state,
        o._n_controls,
        o._unitary_dim,
        o.a,
        o.b,
    )

    outcomes = zeros(o._unitary_dim, 2 * o._unitary_dim)
    for i in 1:o._unitary_dim
        outcomes[i, :] = rand(Multinomial(o.N, p[i, :])) ./ o.N
    end

    u = vec(
        reinterpret(
            Float64, _get_u_from_probabilities(p, o._unitary_dim, o.a, o.b)
        )
    )
    return vcat(state[o._first_indices], u[o._unitary_indices])
end

function observation_space(
    o::UnitaryTomography, state_space::Vector{ClosedInterval{Float64}}
)
    return vcat(
        state_space[o._first_indices],
        state_space[o._unitary_indices .+ o._n_controls .+ 1],
    )
end


"""
    NormalisedObservation(base_function::ObservationFunction, output_dim::Int)

Observation function with output values normalised to unit normal.

Args:
  * `base_function`: Base observation function.
  * `output_dim`: Dimension of the output of the base observation function.

Fields:
  * `base_function`: Base observation function.
  * `observations_mean`: Observations mean.
  * `observations_var`: Observations variance.
  * `count`: The number of seen observations.
"""
struct NormalisedObservation{O <: ObservationFunction} <: ObservationFunction
    base_function::O
    observations_mean::Vector{Float64}
    observations_var::Vector{Float64}
    count::Base.RefValue{Int}
end

function NormalisedObservation(
    base_function::ObservationFunction, output_dim::Int
)
    if base_function isa NormalisedObservation
        throw(
            ArgumentError(
                "Base observation function can't already be the normalised"
                * "observation function!"
            )
        )
    end
    return NormalisedObservation(
        base_function,
        zeros(Float64, output_dim),
        ones(Float64, output_dim),
        Base.RefValue(1),
    )
end

function (n::NormalisedObservation)(state::Vector{Float64})
    observation = n.base_function(state)
    delta_obs = observation .- n.observations_mean
    @. n.observations_mean += delta_obs / (n.count[] + 1)
    @. n.observations_var = (
        n.observations_var * n.count[] / (n.count[] + 1)
        + (delta_obs ^ 2) * n.count[] / (n.count[] + 1) ^ 2
    )
    n.count[] += 1
    return @. (
        (observation - n.observations_mean) / sqrt(n.observations_var + 1e-6)
    )
end

function observation_space(
    o::NormalisedObservation, state_space::Vector{ClosedInterval{Float64}}
)
    return observation_space(o.base_function, state_space)
end


function _get_probabilities_from_state(
    state::Vector{Float64},
    n_controls::Int,
    unitary_dim::Int,
    a::Float64,
    b::Float64,
)
    p = zeros(unitary_dim, 2 * unitary_dim)
    @inbounds @simd for n in 1:unitary_dim
        for i in 1:unitary_dim
            if n == 1
                if i == 1
                    p[1, 1] = a * (
                        state[n_controls + 2] ^ 2 + state[n_controls + 3] ^ 2
                    )
                else
                    p[1, i] = b * (
                        1
                        + 2
                        * (
                            state[n_controls + 2 * i] * state[n_controls + 2]
                            + state[n_controls + 2 * i + 1]
                            * state[n_controls + 3]
                        )
                    )
                    p[1, i + unitary_dim - 1] = b * (
                        1
                        + 2
                        * (
                            state[n_controls + 2 * i + 1]
                            * state[n_controls + 2]
                            - state[n_controls + 2 * i] * state[n_controls + 3]
                        )
                    )
                end
            else
                if i == 1
                    p[n, 1] = (
                        0.5
                        * a
                        * (
                            state[n_controls + 2] ^ 2
                            + state[n_controls + 3] ^ 2
                            + state[n_controls + 2 * unitary_dim * (n - 1) + 2]
                            ^ 2
                            + state[n_controls + 2 * unitary_dim * (n - 1) + 3]
                            ^ 2
                            + 2
                            * (
                                state[n_controls + 2]
                                 * state[
                                    n_controls + 2 * unitary_dim * (n - 1) + 2
                                ]
                                + state[n_controls + 3]
                                * state[
                                    n_controls + 2 * unitary_dim * (n - 1) + 3
                                ]
                            )
                        )
                    )
                else
                    p[n, i] = b * (
                        1
                        + state[n_controls + 2 * i] * state[n_controls + 2]
                        + state[n_controls + 2 * i + 1] * state[n_controls + 3]
                        + state[n_controls + 2 * i]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 2]
                        + state[n_controls + 2 * i + 1]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 3]
                        + state[n_controls + 2 * (i + unitary_dim * (n - 1))]
                        * state[n_controls + 2]
                        + state[
                            n_controls + 2 * (i + unitary_dim * (n - 1)) + 1
                        ]
                        * state[n_controls + 3]
                        + state[n_controls + 2 * (i + unitary_dim * (n - 1))]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 2]
                        + state[
                            n_controls + 2 * (i + unitary_dim * (n - 1)) + 1
                        ]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 3]
                    )
                    p[n, i + unitary_dim - 1] = b * (
                        1
                        + state[n_controls + 2 * i + 1] * state[n_controls + 2]
                        - state[n_controls + 2 * i] * state[n_controls + 3]
                        + state[n_controls + 2 * i + 1]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 2]
                        - state[n_controls + 2 * i]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 3]
                        + state[
                            n_controls + 2 * (i + unitary_dim * (n - 1)) + 1
                        ]
                        * state[n_controls + 2]
                        - state[n_controls + 2 * (i + unitary_dim * (n - 1))]
                        * state[n_controls + 3]
                        + state[
                            n_controls + 2 * (i + unitary_dim * (n - 1)) + 1
                        ]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 2]
                        - state[n_controls + 2 * (i + unitary_dim * (n - 1))]
                        * state[n_controls + 2 * unitary_dim * (n - 1) + 3]
                    )
                end
            end
        end
        p[n, 2 * unitary_dim] = 1 - sum(p[n, 1 : 2 * unitary_dim - 1])
    end
    if any(p .< 0)
        throw(
            ArgumentError(
                "Probabilities must be non-negative. Ensure that `a` and `b`"
                * " are not too large."
            )
        )
    end
    return p
end


function _get_u_from_probabilities(
    p::Matrix{Float64}, unitary_dim::Int, a::Float64, b::Float64
)
    u = zeros(ComplexF64, unitary_dim, unitary_dim)
    psi_out = zeros(ComplexF64, unitary_dim)

    @inbounds for n in 1:unitary_dim
        psi_out[1] = sqrt(p[n, 1] / a)
        for i in 2:unitary_dim
            psi_out[i] = (
                (p[n, i] - b + im * (p[n, i + unitary_dim - 1] - b))
                / (2 * b * psi_out[1])
            )
        end
        if n == 1
            @. u[:, 1] = psi_out
        else
            @. u[:, n] = (2 * psi_out) * $dot(psi_out, u[:, 1]) - u[:, 1]
        end
    end
    if any(isnan, u)
        throw(
            ArgumentError(
                "Probabilities must be consistent with a valid unitary matrix."
            )
        )
    end
    v, _, w = svd(u)  # return closest unitary
    return v * w'
end
