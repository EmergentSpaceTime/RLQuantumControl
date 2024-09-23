"""Abstract callable struct that encodes the "measurement" information the agent
can recieve to decide actions. Custom observations should define an
[`observation_space`]() method.

These callables have the argument signature (excluding an optional RNG):
```math
    \\mathscr{O}(s_{t})\\rightarrow o_{t}
```
"""
abstract type ObservationFunction <: Function end


"""
    FullObservation()

Full state observation including time-to-go, pulse, and evolution operator.
"""
struct FullObservation <: ObservationFunction end

function (::FullObservation)(
    state::Vector{Float64}, ::AbstractRNG = default_rng()
)
    return state
end

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

function (m::MinimalObservation)(
    state::Vector{Float64}, ::AbstractRNG = default_rng()
)
    state[m._observation_indices]
end

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

function (o::ExactTomography)(
    state::Vector{Float64}, ::AbstractRNG = default_rng()
)
    return state[o._observation_indices]
end

function observation_space(
    o::ExactTomography, state_space::Vector{ClosedInterval{Float64}}
)
    return state_space[o._observation_indices]
end


struct UnitaryTomography <: ObservationFunction
    n::Int
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
        n::Int = 10000,
        a::Real = 0.045,
        b::Real = 0.05,
    )

Partial RL state observation sampled from a multinomial distribution of POVM
outcomes with the following POVM and input elements [Baldwin_2014](@cite):
```math
    \\begin{aligned}
        & \\vert\\psi_{0}\\rangle = \\vert 0\\rangle\\\\
        & \\vert\\psi_{n}\\rangle = \\frac{1}{\\sqrt{2}}\\left(
            \\vert 0\\rangle + \\vert n\\rangle
        \\right)\\ \\ \\ \\ n = 1, \\ldots, d - 1
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
        \\right)\\ \\ \\ \\ j = 1, \\ldots, d - 1\\\\
        & \\widetilde{E}_{j} = b\\left(
            1
            + i\\vert j\\rangle\\langle 0\\vert
            - i\\vert 0\\rangle\\langle j\\vert
        \\right)\\ \\ \\ \\ j = 1, \\ldots, d - 1\\\\
        & E_{2d} = 1 - E_{0} - \\sum_{j = 1}^{d - 1} E_{j} + \\widetilde{E}_{j}
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
  * `n`: Number of samples to draw from the multinomial distribution (default:
            10000).
  * `a`: Real positive number to ensure POVM positivity (default: 0.045).
  * `b`: Real positive number to ensure POVM positivity (default: 0.05).

Fields:
  * `n`: Number of samples to draw from the multinomial distribution.
  * `a`: Real positive number to ensure POVM positivity.
  * `b`: Real positive number to ensure POVM positivity.
"""
function UnitaryTomography(
    n_controls::Int,
    observation_type::String,
    unitary_dim::Int,
    include_pulse::Bool,
    cols::Tuple{Vararg{Int}} = ();
    n::Int = 10000,
    a::Real = 0.045,
    b::Real = 0.05,
)
    n_controls < 1 && throw(ArgumentError("`n_controls` must be >= 1."))
    if (a <= 0) | (b <= 0)
        throw(ArgumentError("`a` and `b` must be positive real numbers."))
    end
    n <= 0 && throw(ArgumentError("`N` must be a positive integer."))
    if n < 1000
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
        n, a, b, n_controls, unitary_dim, _first_indices, _unitary_indices
    )
end

function (o::UnitaryTomography)(
    state::Vector{Float64}, rng::AbstractRNG = default_rng()
)
    p = _get_probabilities_from_state(
        state,
        o._n_controls,
        o._unitary_dim,
        o.a,
        o.b,
    )

    outcomes = zeros(o._unitary_dim, 2 * o._unitary_dim)
    for i in 1:o._unitary_dim
        outcomes[i, :] = rand(rng, Multinomial(o.n, p[i, :])) ./ o.n
    end

    u = vec(
        reinterpret(
            Float64,
            _get_u_from_probabilities(outcomes, o._unitary_dim, o.a, o.b),
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
        Base.RefValue(0),
    )
end

function (o::NormalisedObservation)(state::Vector{Float64})
    obs = o.base_function(state)
    delta_obs = obs .- o.observations_mean  # Update mean.
    @. o.observations_mean += delta_obs / (o.count[] + 1)
    delta_obs_new = obs .- o.observations_mean  # Update variance.
    @. o.observations_var = (
        o.count[] * o.observations_var / (o.count[] + 1)
        + delta_obs * delta_obs_new / (o.count[] + 1)
    )
    o.count[] += 1  # Update count.
    return @. (
        (obs - o.observations_mean) / sqrt(o.observations_var + 1e-6)
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
                / (2 * b * psi_out[1] + 1e-6)
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
    v, _, w = svd(u)  # Returns closest unitary.
    return v * w'
end


function _get_povm_set(a::Real, b::Real, d::Int)
    states = I(d)

    povms = zeros(ComplexF64, d, d, 2d)
    povms[:, :, 1] .= a .* (states[:, 1] * states[:, 1]')
    for i in 2:d
        povms[:, :, i] .= b .* (
                states
                .+ states[:, 1] * states[:, i]'
                .+ states[:, i] * states[:, 1]'
            )
        povms[:, :, i + d - 1] .= b .* (
                states
                .+ im .* states[:, 1] * states[:, i]'
                .- im .* states[:, i] * states[:, 1]'
            )
    end
    povms[:, :, end] .= (
        states .- dropdims(sum(povms[:, :, 1 : end - 1]; dims=3); dims=3)
    )
    return povms
end


_lowest_eigval(m::Matrix{ComplexF64}) = minimum(eigvals(Hermitian(m)))


function _tomography_bisection(
    unitary_dim::Int, b::Float64, n_max::Int; atol=1e-6
)
    x_min, x_max = 1e-3, 1 - 1e-3

    f_x_min = _lowest_eigval(_get_povm_set(x_min, b, unitary_dim)[:, :, end])
    f_x_max = _lowest_eigval(_get_povm_set(x_max, b, unitary_dim)[:, :, end])
    if ((f_x_min < 0) & (f_x_max < 0)) | ((f_x_min > 0) & (f_x_max > 0))
        throw(DomainError((f_x_min, f_x_max), "Root is not in the interval."))
    end

    i = 0
    while i <= n_max
        x_mid = (x_min + x_max) / 2
        f_x_mid = _lowest_eigval(
            _get_povm_set(x_mid, b, unitary_dim)[:, :, end]
        )
        if (isapprox(f_x_mid, 0)) | ((x_max - x_min) / 2 < atol)
            return x_mid, f_x_mid
        end
        if sign(f_x_mid) == sign(f_x_min)
            x_min = x_mid
        else
            x_max = x_mid
        end
        i += 1
    end
    return "Solution did not converge."
end
